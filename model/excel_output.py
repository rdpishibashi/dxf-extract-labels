"""Excel 出力モジュール

通常モード（ラベル抽出）・領域付きモード（矩形領域付きラベル抽出）の
Excel ファイル生成を担う。集計ロジックは `ref_designator.py`
（機器符号抽出パイプライン）に実装されている。

出力ファイルに記録するラベル・機器符号は `normalize_width()` で半角へ統一する
（正規化は集計前に呼び出し元〈`ref_designator.py`〉側で行われる）。

v3.0.0（2026-08）: 「機器符号（候補）以外も抽出」「図面番号・タイトル・
サブタイトル抽出」の両オプションを廃止したことに伴う出力刷新。
`create_excel_output()`（前者オプション専用）を削除し、通常モード・領域モードの
2経路に統一。Summary シートは両モード共通の8列、`Total` シートは
`抽出ラベル一覧` に改称、各ラベル一覧系シートに「機器符号候補」列を追加、
Summary 以外の全シートにオートフィルタを設定した。
"""
import os
import pandas as pd
from io import BytesIO
from collections import Counter

from .common_utils import normalize_width
from .ref_designator import designator_mark
from .region_detector import build_region_label_summary
from .terminal_detector import build_terminal_rows
from .assembly_compare import COMPARE_COLUMNS, COMPARE_SHEET_NAME, ROW_STYLE_COLORS, row_style

# Summary シートの列構成（v3.0.0、通常モード・領域モード共通）。
# 通常モードでは「検出領域数」に常に 0 を入れる。
SUMMARY_COLUMNS = (
    'ファイル名', '総抽出ラベル数', '機器符号候補ラベル数', 'ページ数',
    '検出領域数', '図番', 'タイトル', 'サブタイトル',
)


def _add_standard_formats(workbook):
    """全シート共通のヘッダー書式・ハイパーリンク書式を返す。"""
    header_format = workbook.add_format({'bold': True, 'bg_color': '#D3D3D3', 'border': 1})
    link_format = workbook.add_format({'color': 'blue', 'underline': 1})
    return header_format, link_format


def _write_header_row(ws, columns, header_format):
    """ヘッダー行（1行目）の各セルに書式を適用し、先頭行を固定する。"""
    for col_num, value in enumerate(columns):
        ws.write(0, col_num, value, header_format)
    ws.freeze_panes(1, 0)


def _write_summary_hyperlinks(ws, filenames, link_format, col=0, start_row=1):
    """Summary シートの各行から、対応するファイルシート（先頭31文字）への
    内部ハイパーリンクを書き込む。"""
    for row_num, fname in enumerate(filenames, start=start_row):
        sheet_name = os.path.splitext(fname)[0][:31]
        ws.write_url(row_num, col, f"internal:'{sheet_name}'!A1", link_format, fname)


def _write_label_count_sheet(writer, sheet_name, rows, header_format,
                              columns=('機器符号候補', 'ラベル', '個数'), col_widths=(12, 25, 10),
                              skip_if_empty=True):
    """`[{列名: 値, ...}, ...]` 形式の行データを、ヘッダー書式・オートフィルタ付きで
    シートに書き込む（抽出ラベル一覧・各ファイルのラベル一覧シート共通の書式）。
    `rows` が空の場合、`skip_if_empty=True`（既定）なら何もしない（シートを作らない）。
    `skip_if_empty=False` はヘッダー行のみの空シートを作る（`create_region_excel_output`
    の各ファイルシートが、領域内ラベルが0件でもシート自体は必ず作る従来挙動のため）。

    Summary シート以外の全シートにオートフィルタを設定する方針（v3.0.0）の
    実体はここに集約している。Summary シート自体はこの関数を使わず個別に
    書き込むため、この関数からは常にフィルタが付く。
    """
    if not rows and skip_if_empty:
        return None
    df = pd.DataFrame(rows, columns=list(columns))
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]
    _write_header_row(ws, columns, header_format)
    for i, width in enumerate(col_widths):
        letter = chr(ord('A') + i)
        ws.set_column(f'{letter}:{letter}', width)
    ws.autofilter(0, 0, len(df), len(columns) - 1)
    return ws


def _accumulate_total_counts(total_counter, label_drawing_numbers, label_counts, ident):
    """(ラベル, 個数) のペア列を抽出ラベル一覧シート集計用アキュムレータに反映する。"""
    for lbl, count in label_counts:
        total_counter[lbl] += count
        label_drawing_numbers.setdefault(lbl, set()).add(ident)


def _build_total_rows(total_counter, label_drawing_numbers, master_index=None):
    """抽出ラベル一覧（旧 Total）シート用の行データ（ラベル昇順、図番はABC順で
    カンマ結合）を返す。「機器符号候補」列は `designator_mark()` による判定
    （v3.0.0で追加、「ラベル」列の直左に配置。v3.10.0で'DEF'/'CAN'/Noneの
    3値に変更）。このシートはファイル横断でラベルを再集計するため、各ファイルの
    `rows` に既にある「機器符号候補」値は使わず、ラベル文字列から改めて
    判定し直す（`master_index` は呼び出し元から明示的に渡す必要がある）。"""
    return [
        {'機器符号候補': designator_mark(lbl, master_index),
         'ラベル': lbl, '個数': total_counter[lbl],
         '図番': ', '.join(sorted(label_drawing_numbers.get(lbl, ())))}
        for lbl in sorted(total_counter.keys())
    ]


def _write_terminal_sheet(writer, terminal_results, header_format):
    """「TB List」シート（端子台・端子No.・図番）を書き込む。

    `terminal_results` は {filename: analyze_dxf_terminals()結果, ...}。
    None または対象ファイルが1件も無い場合は何もしない（シートを作らない）。
    """
    if not terminal_results:
        return None
    rows = build_terminal_rows(terminal_results)
    return _write_label_count_sheet(
        writer, 'TB List', rows, header_format,
        columns=('端子台', '端子No.', '図番'), col_widths=(25, 40, 20))


def _write_assembly_compare_sheet(writer, workbook, assembly_compare, header_format):
    """「結線図-組立図比較」シートを書き込む（「端子一覧を抽出」ON時のTB List
    直後・各ファイルシートの直前に配置、v3.6.0）。

    `assembly_compare` は `model/assembly_compare.compare_assembly()` の戻り値
    （{'b_file','rows','summary'}）。None または `rows` が空の場合は何もしない
    （シートを作らない）。

    行ごとに区分（両方/A のみ/B のみ）と個数一致で色分けするため（`row_style()`）、
    `_write_label_count_sheet()`（pandas経由の一括書き込み）は使わず、
    `領域別ラベル一覧` と同じ方式でワークシートを直接構築する。

    セルの罫線は他の全シート（`抽出ラベル一覧`・各ファイルシート等）と同様に
    デフォルトのまま（罫線なし）とする。ヘッダー行のみ `header_format`
    （太字・背景色・罫線）を適用する点も他シートと共通。データ行には色分け
    以外の書式を加えない（2026-09-13 ユーザー指摘: 移植元 `DXF-label-compare`
    は全データセルに `border: 1` を付けていたが、本プロジェクトの他シートは
    データ行に罫線を付けない方針のため、そちらには合わせない）。
    """
    if not assembly_compare or not assembly_compare.get('rows'):
        return None
    rows = assembly_compare['rows']

    ws = workbook.add_worksheet(COMPARE_SHEET_NAME)
    for col_idx, col_name in enumerate(COMPARE_COLUMNS):
        ws.write(0, col_idx, col_name, header_format)

    row_formats = {
        style_key: workbook.add_format(color) if color else None
        for style_key, color in ROW_STYLE_COLORS.items()
    }

    for row_idx, r in enumerate(rows, start=1):
        fmt = row_formats[row_style(r['比較結果'], r['A 合計出現数'], r['B 出現数'])]
        for col_idx, col_name in enumerate(COMPARE_COLUMNS):
            ws.write(row_idx, col_idx, r[col_name], fmt)

    ws.set_column('A:A', 12)
    ws.set_column('B:B', 25)
    ws.set_column('C:E', 12)
    ws.set_column('F:F', 30)
    ws.set_column('G:G', 12)
    ws.freeze_panes(1, 0)
    ws.autofilter(0, 0, len(rows), len(COMPARE_COLUMNS) - 1)
    return ws


def _build_summary_row(fname, total_in_frame, ref_designator_count, frames,
                        regions_detected, drawing_number, title, subtitle):
    """Summary シート1行分（`SUMMARY_COLUMNS` の8列）を構築する（v3.0.0、
    通常モード・領域モード共通の列構成。通常モードは regions_detected=0 を渡す）。
    """
    return {
        'ファイル名': fname,
        '総抽出ラベル数': total_in_frame,
        '機器符号候補ラベル数': ref_designator_count,
        'ページ数': frames,
        '検出領域数': regions_detected,
        '図番': drawing_number or '',
        'タイトル': title or '',
        'サブタイトル': subtitle or '',
    }


def create_ref_designator_excel_output(results, sort_option, terminal_results=None,
                                        assembly_compare=None, master_index=None):
    """通常モードの Excel を生成する。

    `results`: {fname: {...}}（キーは各値の意味）
      'rows': [{'機器符号候補':str|None,'ラベル':str,'個数':int}]  枠内・情報枠外の全ラベル
      'total_in_frame': int  図面枠内の総ラベル数
      'ref_designator_count': int  機器符号候補ラベル数（rowsのうち機器符号候補が
          'DEF'/'CAN'〈v3.10.0で3値化。旧'Y'に相当〉の個数合計）
      'frames': int  ページ数
      'warning': str | None
      'main_drawing_number' / 'source_drawing_number' / 'title' / 'subtitle':
          図番・タイトル・サブタイトル（v3.0.0で常に抽出・常に値が入る）

    'rows' はラベルの正規化（NFKC・前後空白除去）済みで渡される想定
    （`model/ref_designator.py` 側で実施済み。'機器符号候補'値も
    `build_ref_designator_final()` 等が `master_index` 込みで計算済みの
    ものをそのまま使う——各ファイルシートはこの値を再計算しない）。

    `master_index`（`model/designator_master.DesignatorMasterIndex` または
    None、v3.10.0新設）: `抽出ラベル一覧` シートはファイル横断でラベルを
    再集計するため、`_build_total_rows()` にこの引数を明示的に渡す必要がある
    （渡し漏れるとこのシートだけDEFが出ない）。

    `assembly_compare`: `model/assembly_compare.compare_assembly()` の戻り値、
    または None（「結線図-組立図を比較」OFF時・比較対象が特定できない場合、
    v3.6.0）。指定時は「TB List」直後・各ファイルシートの直前に
    「結線図-組立図比較」シートを追加する。

    Summary シート・各ファイルシートともファイル名昇順で出力する。
    """
    output = BytesIO()

    sorted_items = sorted(results.items(), key=lambda kv: kv[0])

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        header_format, link_format = _add_standard_formats(workbook)

        summary_data = []
        total_counter = Counter()
        label_drawing_numbers = {}
        for fname, data in sorted_items:
            summary_data.append(_build_summary_row(
                fname,
                data.get('total_in_frame', 0),
                data.get('ref_designator_count', 0),
                data.get('frames', 0),
                0,
                data.get('main_drawing_number'),
                data.get('title'),
                data.get('subtitle'),
            ))

            ident = data.get('main_drawing_number') or os.path.splitext(fname)[0]
            _accumulate_total_counts(
                total_counter, label_drawing_numbers,
                ((r['ラベル'], r['個数']) for r in data.get('rows', [])), ident)

        summary_df = pd.DataFrame(summary_data, columns=list(SUMMARY_COLUMNS))
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        summary_ws = writer.sheets['Summary']
        _write_header_row(summary_ws, summary_df.columns.values, header_format)
        _write_summary_hyperlinks(summary_ws, [fname for fname, _data in sorted_items], link_format)
        summary_ws.set_column('A:A', 28)

        total_data = _build_total_rows(total_counter, label_drawing_numbers, master_index=master_index)
        _write_label_count_sheet(writer, '抽出ラベル一覧', total_data, header_format,
                                 columns=('機器符号候補', 'ラベル', '個数', '図番'),
                                 col_widths=(12, 25, 10, 30))

        _write_terminal_sheet(writer, terminal_results, header_format)
        _write_assembly_compare_sheet(writer, workbook, assembly_compare, header_format)

        for fname, data in sorted_items:
            sheet_name = os.path.splitext(fname)[0][:31]
            rows = list(data.get('rows', []))
            if sort_option == 'asc':
                rows.sort(key=lambda r: r['ラベル'])
            elif sort_option == 'desc':
                rows.sort(key=lambda r: r['ラベル'], reverse=True)
            _write_label_count_sheet(writer, sheet_name, rows, header_format,
                                     columns=('機器符号候補', 'ラベル', '個数'),
                                     col_widths=(12, 25, 10))

    output.seek(0)
    return output.getvalue()


def create_region_excel_output(region_results, terminal_results=None, assembly_compare=None,
                                master_index=None):
    """矩形領域抽出結果の Excel を生成する。各ファイルシートに『領域』列を付与する。

    `master_index`（`model/designator_master.DesignatorMasterIndex` または
    None、v3.10.0新設）: `抽出ラベル一覧`・`領域別ラベル一覧` の2シートは
    ファイル横断でラベルを再集計するため、この引数を明示的に渡す必要がある
    （渡し漏れるとこの2シートだけDEFが出ない。各ファイルシート・`領域一覧`
    シートの値は呼び出し元で `master_index` 込みで計算済みの `region_results`
    をそのまま使うため、ここでの引数とは別に正しく反映される）。

    `assembly_compare`: `create_ref_designator_excel_output()` と同じ
    （`model/assembly_compare.compare_assembly()` の戻り値、または None）。
    指定時は「TB List」直後・各ファイルシートの直前に「結線図-組立図比較」
    シートを追加する（v3.6.0）。

    Summary シート・各ファイルシートともファイル名昇順で出力する
    （region_results をソート済み dict に差し替えることで、領域別ラベル
    一覧のファイル列順にも一括で反映される）。

    Summary シートは通常モードと共通の8列構成（`SUMMARY_COLUMNS`）。図番・
    タイトル・サブタイトルは v3.0.0で常に抽出・常に列に値が入る。

    `Summary` 直後に `抽出ラベル一覧`（旧 `Total`）シート（全ファイル横断の
    機器符号候補・ラベル・個数・図番のユニーク集計、v1.9.19）を追加する。
    通常モードと同じ `_accumulate_total_counts`/`_build_total_rows` を再利用し、
    各ファイルの `rows`（既に半角正規化・集計済み）を `drawing_number`
    （未抽出時はファイル名）で紐付けて集計する。

    `領域一覧` シートは、検出した全領域（確定・未確定を問わない）を
    `ファイル名×領域名` でマージ・合算して記載する（v3.0.0、`ページ No.` 列は
    廃止し、面積率・領域内ラベル数・領域内機器符号候補数をページ横断で合算する。
    データは各ファイルの `region_rows`〈`ref_designator.build_all_regions_summary()`
    参照〉から得る）。
    """
    region_results = dict(sorted(region_results.items(), key=lambda kv: kv[0]))
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        header_format, link_format = _add_standard_formats(workbook)

        summary_data = []
        for fname, data in region_results.items():
            summary_data.append(_build_summary_row(
                fname,
                data.get('total_in_frame', 0),
                data.get('ref_designator_count', 0),
                data.get('frames', 0),
                data.get('regions_detected', 0),
                data.get('drawing_number'),
                data.get('title'),
                data.get('subtitle'),
            ))
        summary_df = pd.DataFrame(summary_data, columns=list(SUMMARY_COLUMNS))
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        summary_ws = writer.sheets['Summary']
        _write_header_row(summary_ws, summary_df.columns.values, header_format)
        _write_summary_hyperlinks(summary_ws, list(region_results.keys()), link_format)
        summary_ws.set_column('A:A', 28)

        total_counter = Counter()
        label_drawing_numbers = {}
        for fname, data in region_results.items():
            ident = data.get('drawing_number') or os.path.splitext(fname)[0]
            _accumulate_total_counts(
                total_counter, label_drawing_numbers,
                ((r['ラベル'], r['個数']) for r in data.get('rows', [])), ident)
        total_data = _build_total_rows(total_counter, label_drawing_numbers, master_index=master_index)
        _write_label_count_sheet(writer, '抽出ラベル一覧', total_data, header_format,
                                 columns=('機器符号候補', 'ラベル', '個数', '図番'),
                                 col_widths=(12, 25, 10, 30))

        # 領域一覧: 検出した全領域（確定・未確定を問わない）を、ファイル名×領域名で
        # マージ・合算して記載する（v3.0.0）。
        merged_regions = {}
        region_order = []
        for fname, data in region_results.items():
            for r in data.get('region_rows', []):
                key = (fname, r['name'])
                if key not in merged_regions:
                    merged_regions[key] = {'area_pct': 0.0, 'label_count': 0, 'ref_designator_count': 0}
                    region_order.append(key)
                agg = merged_regions[key]
                agg['area_pct'] += r['area_pct']
                agg['label_count'] += r['label_count']
                agg['ref_designator_count'] += r['ref_designator_count']

        reg_rows = [
            {
                'ファイル名': fname,
                '領域名': name,
                '面積率[%]': round(merged_regions[(fname, name)]['area_pct'], 1),
                '領域内ラベル数': merged_regions[(fname, name)]['label_count'],
                '領域内機器符号候補数': merged_regions[(fname, name)]['ref_designator_count'],
            }
            for (fname, name) in region_order
        ]
        if reg_rows:
            reg_df = pd.DataFrame(reg_rows)
            reg_df.to_excel(writer, sheet_name='領域一覧', index=False)
            reg_ws = writer.sheets['領域一覧']
            _write_header_row(reg_ws, reg_df.columns.values, header_format)
            reg_ws.set_column('A:A', 28)
            reg_ws.set_column('B:B', 30)
            reg_ws.autofilter(0, 0, len(reg_df), len(reg_df.columns) - 1)

        # 領域別ラベル一覧: 領域名ごとに全ファイル横断でラベルを集計する
        # （領域名は同名なら複数ファイルにまたがって合算。詳細は
        # build_region_label_summary の docstring 参照。確定済み名称のみ対象
        # ——未確定領域の候補名はここには含めない。意図的な非対称、v3.0.0）。
        files, label_summary_rows = build_region_label_summary(region_results)
        if label_summary_rows:
            lbl_ws = workbook.add_worksheet('領域別ラベル一覧')
            lbl_ws.write(0, 0, '領域名', header_format)
            lbl_ws.write(0, 1, '機器符号候補', header_format)
            lbl_ws.write(0, 2, 'ラベル', header_format)
            lbl_ws.write(0, 3, '合計個数', header_format)
            col = 4
            for _fname, _ident in files:
                lbl_ws.write(0, col, '図番', header_format)
                lbl_ws.write(0, col + 1, '個数', header_format)
                col += 2

            for row_num, r in enumerate(label_summary_rows, start=1):
                lbl_ws.write(row_num, 0, r['領域名'])
                lbl_ws.write(row_num, 1, designator_mark(r['ラベル'], master_index))
                lbl_ws.write(row_num, 2, r['ラベル'])
                lbl_ws.write(row_num, 3, r['合計個数'])
                col = 4
                for fname, ident in files:
                    lbl_ws.write(row_num, col, ident)
                    lbl_ws.write(row_num, col + 1, r['per_file'].get(fname, 0))
                    col += 2

            lbl_ws.set_column('A:A', 28)
            lbl_ws.set_column('B:B', 12)
            lbl_ws.set_column('C:C', 25)
            lbl_ws.set_column(3, 3, 10)
            lbl_ws.set_column(4, col - 1, 16)
            lbl_ws.freeze_panes(1, 0)
            lbl_ws.autofilter(0, 0, len(label_summary_rows), col - 1)

        _write_terminal_sheet(writer, terminal_results, header_format)
        _write_assembly_compare_sheet(writer, workbook, assembly_compare, header_format)

        for fname, data in region_results.items():
            sheet_name = os.path.splitext(fname)[0][:31]
            _write_label_count_sheet(
                writer, sheet_name, data['rows'], header_format,
                columns=('機器符号候補', 'ラベル', '個数', '領域'), col_widths=(12, 25, 8, 40),
                skip_if_empty=False)

    output.seek(0)
    return output.getvalue()
