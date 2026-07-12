"""Excel 出力モジュール

通常モード（ラベル抽出）・領域付きモード（矩形領域付きラベル抽出）の
Excel ファイル生成を担う。集計ロジック（build_region_results）は
region_detector モジュールに実装されている。

出力ファイルに記録するラベル・機器符号は `normalize_width()` で半角へ統一する
（領域付きモードの正規化は build_region_results 側で集計前に行われる）。
"""
import os
import pandas as pd
from io import BytesIO
from collections import Counter

from .common_utils import normalize_width
from .region_detector import build_region_label_summary


def _add_standard_formats(workbook):
    """全シート共通のヘッダー書式・ハイパーリンク書式を返す。"""
    header_format = workbook.add_format({'bold': True, 'bg_color': '#D3D3D3', 'border': 1})
    link_format = workbook.add_format({'color': 'blue', 'underline': 1})
    return header_format, link_format


def _write_header_row(ws, columns, header_format):
    """ヘッダー行（1行目）の各セルに書式を適用する。"""
    for col_num, value in enumerate(columns):
        ws.write(0, col_num, value, header_format)


def _write_summary_hyperlinks(ws, filenames, link_format, col=0, start_row=1):
    """Summary シートの各行から、対応するファイルシート（先頭31文字）への
    内部ハイパーリンクを書き込む。"""
    for row_num, fname in enumerate(filenames, start=start_row):
        sheet_name = os.path.splitext(fname)[0][:31]
        ws.write_url(row_num, col, f"internal:'{sheet_name}'!A1", link_format, fname)


def _write_label_count_sheet(writer, sheet_name, rows, header_format,
                              columns=('ラベル', '個数'), col_widths=(25, 10),
                              skip_if_empty=True):
    """`[{列名: 値, ...}, ...]` 形式の行データを、ヘッダー書式付きでシートに
    書き込む（Total・各ファイルのラベル一覧シート共通の書式）。`rows` が空の
    場合、`skip_if_empty=True`（既定）なら何もしない（シートを作らない）。
    `skip_if_empty=False` はヘッダー行のみの空シートを作る（`create_region_excel_output`
    の各ファイルシートが、領域内ラベルが0件でもシート自体は必ず作る従来挙動のため）。"""
    if not rows and skip_if_empty:
        return None
    df = pd.DataFrame(rows, columns=list(columns))
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]
    _write_header_row(ws, columns, header_format)
    for i, width in enumerate(col_widths):
        letter = chr(ord('A') + i)
        ws.set_column(f'{letter}:{letter}', width)
    return ws


def create_excel_output(results, sort_option):
    """通常モード（「機器符号（候補）以外も抽出」ON）の Excel を生成する。

    ラベルは半角へ正規化してから集計するため、図面上の表記が半角/全角どちら
    でも同じ語は同じ行（同じ個数）にまとまる。フィルタなし（全ラベル）で
    抽出された `results` をそのまま集計する。

    Summary シート・各ファイルシートともファイル名昇順で出力する。
    """
    output = BytesIO()

    sorted_items = sorted(
        results.items(),
        key=lambda kv: kv[1][1].get('filename', os.path.basename(kv[0])),
    )

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        header_format, link_format = _add_standard_formats(workbook)

        summary_data = []
        total_counter = Counter()

        for file_path, (labels, info) in sorted_items:
            filename = info.get('filename', os.path.basename(file_path))
            summary_data.append({
                'ファイル名': filename,
                '総ラベル数': info.get('final_count', 0),
                '図番': info.get('main_drawing_number', ''),
                '流用元図番': info.get('source_drawing_number', ''),
                'タイトル': info.get('title', ''),
                'サブタイトル': info.get('subtitle', '')
            })

            total_counter.update(Counter(normalize_width(l) for l in labels))

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        summary_worksheet = writer.sheets['Summary']
        _write_header_row(summary_worksheet, summary_df.columns.values, header_format)

        filenames = [info.get('filename', os.path.basename(fp))
                    for fp, (_labels, info) in sorted_items]
        _write_summary_hyperlinks(summary_worksheet, filenames, link_format)

        total_data = [
            {'ラベル': lbl, '個数': total_counter[lbl]}
            for lbl in sorted(total_counter.keys())
        ]
        _write_label_count_sheet(writer, 'Total', total_data, header_format)

        for file_path, (labels, info) in sorted_items:
            filename = info.get('filename', os.path.basename(file_path))
            sheet_name = os.path.splitext(filename)[0][:31]

            counter = Counter(normalize_width(l) for l in labels)
            label_data = [
                {'ラベル': lbl, '個数': counter[lbl]}
                for lbl in sorted(counter.keys())
            ]
            _write_label_count_sheet(writer, sheet_name, label_data, header_format)

    output.seek(0)
    return output.getvalue()


def create_ref_designator_excel_output(results, sort_option):
    """通常モード（既定＝「機器符号（候補）以外も抽出」OFF）の Excel を生成する。

    `results`: {fname: {...}}（キーは各値の意味）
      'rows': [{'ラベル':str,'個数':int}]  ユーザーが確定した機器符号（候補）
      'total_in_frame': int  図面枠内の総ラベル数
      'unclassified_count': int  未採用のまま残った未確定ラベルの種類数
      'warning': str | None
      'main_drawing_number' / 'source_drawing_number' / 'title' / 'subtitle':
          図番・タイトル抽出オプション有効時のみ値が入る

    'rows' はラベルの正規化（NFKC・前後空白除去）済みで渡される想定
    （`utils/ref_designator.py` 側で実施済み）。

    Summary シート・各ファイルシートともファイル名昇順で出力する。
    """
    output = BytesIO()

    sorted_items = sorted(results.items(), key=lambda kv: kv[0])

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        header_format, link_format = _add_standard_formats(workbook)

        summary_data = []
        total_counter = Counter()
        for fname, data in sorted_items:
            row = {
                'ファイル名': fname,
                '図面枠内ラベル数': data.get('total_in_frame', 0),
                '機器符号（候補）数': sum(r['個数'] for r in data.get('rows', [])),
                '未確定ラベル数（未採用）': data.get('unclassified_count', 0),
            }
            if data.get('main_drawing_number') is not None or data.get('title') is not None:
                row['図番'] = data.get('main_drawing_number', '') or ''
                row['流用元図番'] = data.get('source_drawing_number', '') or ''
                row['タイトル'] = data.get('title', '') or ''
                row['サブタイトル'] = data.get('subtitle', '') or ''
            summary_data.append(row)
            for r in data.get('rows', []):
                total_counter[r['ラベル']] += r['個数']

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        summary_ws = writer.sheets['Summary']
        _write_header_row(summary_ws, summary_df.columns.values, header_format)
        _write_summary_hyperlinks(summary_ws, [fname for fname, _data in sorted_items], link_format)
        summary_ws.set_column('A:A', 28)

        total_data = [{'ラベル': lbl, '個数': total_counter[lbl]} for lbl in sorted(total_counter.keys())]
        _write_label_count_sheet(writer, 'Total', total_data, header_format)

        for fname, data in sorted_items:
            sheet_name = os.path.splitext(fname)[0][:31]
            rows = list(data.get('rows', []))
            if sort_option == 'asc':
                rows.sort(key=lambda r: r['ラベル'])
            elif sort_option == 'desc':
                rows.sort(key=lambda r: r['ラベル'], reverse=True)
            _write_label_count_sheet(writer, sheet_name, rows, header_format)

    output.seek(0)
    return output.getvalue()


def create_region_excel_output(region_results):
    """矩形領域抽出結果の Excel を生成する。各ファイルシートに『領域』列を付与する。

    Summary シート・各ファイルシートともファイル名昇順で出力する
    （region_results をソート済み dict に差し替えることで、領域別ラベル
    一覧のファイル列順にも一括で反映される）。
    """
    region_results = dict(sorted(region_results.items(), key=lambda kv: kv[0]))
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        header_format, link_format = _add_standard_formats(workbook)

        summary_data = []
        for fname, data in region_results.items():
            summary_data.append({
                'ファイル名': fname,
                '総抽出ラベル数': data.get('total_in_frame', 0),
                '除外ラベル数': data.get('filtered_count', 0),
                '最終ラベル数': data.get('final_count', 0),
                '図面枠数': data['frames'],
                '検出領域数': data['regions_detected'],
                '確定領域数': data['regions_named'],
                '領域内ラベル数': data['in_region_count'],
            })
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        summary_ws = writer.sheets['Summary']
        _write_header_row(summary_ws, summary_df.columns.values, header_format)
        _write_summary_hyperlinks(summary_ws, list(region_results.keys()), link_format)
        summary_ws.set_column('A:A', 28)

        reg_rows = []
        for fname, data in region_results.items():
            for r in data['named']:
                reg_rows.append({
                    'ファイル名': fname,
                    'ページ No.': r['frame'] + 1,
                    '領域名': r['name'],
                    '面積率[%]': round(r['area_pct'], 1),
                    '領域内ラベル数': r['label_count'],
                })
        if reg_rows:
            reg_df = pd.DataFrame(reg_rows)
            reg_df.to_excel(writer, sheet_name='領域一覧', index=False)
            reg_ws = writer.sheets['領域一覧']
            _write_header_row(reg_ws, reg_df.columns.values, header_format)
            reg_ws.set_column('A:A', 28)
            reg_ws.set_column('C:C', 22)

        # 領域別ラベル一覧: 領域名ごとに全ファイル横断でラベルを集計する
        # （領域名は同名なら複数ファイルにまたがって合算。詳細は
        # build_region_label_summary の docstring 参照）。
        files, label_summary_rows = build_region_label_summary(region_results)
        if label_summary_rows:
            lbl_ws = workbook.add_worksheet('領域別ラベル一覧')
            lbl_ws.write(0, 0, '領域名', header_format)
            lbl_ws.write(0, 1, 'ラベル', header_format)
            lbl_ws.write(0, 2, '合計個数', header_format)
            col = 3
            for _fname, _ident in files:
                lbl_ws.write(0, col, '図番', header_format)
                lbl_ws.write(0, col + 1, '個数', header_format)
                col += 2

            for row_num, r in enumerate(label_summary_rows, start=1):
                lbl_ws.write(row_num, 0, r['領域名'])
                lbl_ws.write(row_num, 1, r['ラベル'])
                lbl_ws.write(row_num, 2, r['合計個数'])
                col = 3
                for fname, ident in files:
                    lbl_ws.write(row_num, col, ident)
                    lbl_ws.write(row_num, col + 1, r['per_file'].get(fname, 0))
                    col += 2

            lbl_ws.set_column('A:A', 28)
            lbl_ws.set_column('B:B', 25)
            lbl_ws.set_column(2, 2, 10)
            lbl_ws.set_column(3, col - 1, 16)

        for fname, data in region_results.items():
            sheet_name = os.path.splitext(fname)[0][:31]
            _write_label_count_sheet(
                writer, sheet_name, data['rows'], header_format,
                columns=('ラベル', '個数', '領域'), col_widths=(25, 8, 40),
                skip_if_empty=False)

    output.seek(0)
    return output.getvalue()
