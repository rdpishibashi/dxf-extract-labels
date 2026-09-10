"""create_ref_designator_excel_output / create_region_excel_output のスモークテスト。

Excel 生成関数が最小の入力で例外なく完走し、
正しいシート構成の Excel を返すことを検証する。
ロジックの詳細（書式・列幅）ではなく「動作する・壊れていない」を確認する。

v3.0.0（2026-08）: 「機器符号（候補）以外も抽出」オプション廃止に伴い
`create_excel_output()` が削除されたため、そのテスト（`test_create_excel_output_*`）
も削除した。あわせて `Total` シートは `抽出ラベル一覧` に改称、各ラベル一覧系
シートに「機器符号候補」列（'Y' または None）を追加、Summary シートは通常/領域
両モード共通の8列固定（`図番`/`タイトル`/`サブタイトル`列の条件付き追加は廃止）、
`領域一覧` は `ページ No.` 列を廃止し `ファイル名×領域名` でマージ・合算する
仕様に変更されている。`region_detector.build_region_results()` に依存していた
テスト（`test_build_region_results_*`）は、対応する動作が
`ref_designator.build_named_regions()` に移った関数レベルのテストとして
`tests/regression/test_ref_designator.py` に移設した。
"""
import sys
import os
from io import BytesIO

import openpyxl
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model.excel_output import (  # noqa: E402
    create_ref_designator_excel_output, create_region_excel_output,
)


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _load_wb(data: bytes):
    return openpyxl.load_workbook(BytesIO(data))


# ---------------------------------------------------------------------------
# create_ref_designator_excel_output
# ---------------------------------------------------------------------------

def _make_ref_designator_results():
    return {
        'fileA.dxf': {
            'rows': [
                {'機器符号候補': 'Y', 'ラベル': 'CN1', '個数': 2},
                {'機器符号候補': 'Y', 'ラベル': 'R10', '個数': 1},
            ],
            'total_in_frame': 10,
            'ref_designator_count': 3,
            'frames': 1,
            'main_drawing_number': 'EE1234-500-01A',
            'source_drawing_number': 'EE0001-500-01A',
            'title': 'TITLE A',
            'subtitle': 'SUB A',
        },
        'fileB.dxf': {
            'rows': [],
            'total_in_frame': 5,
            'ref_designator_count': 0,
            'frames': 1,
            'main_drawing_number': None,
            'source_drawing_number': None,
            'title': None,
            'subtitle': None,
        },
    }


def test_create_ref_designator_excel_output_returns_bytes():
    data = create_ref_designator_excel_output(_make_ref_designator_results(), 'asc')
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_create_ref_designator_excel_output_sheet_and_values():
    """rows が空のファイルシート（fileB）はスキップする従来挙動を固定する。
    Summary は通常モードでも常に8列（検出領域数は0）。"""
    wb = _load_wb(create_ref_designator_excel_output(_make_ref_designator_results(), 'asc'))
    assert wb.sheetnames == ['Summary', '抽出ラベル一覧', 'fileA']

    summary = wb['Summary']
    header = [c.value for c in summary[1]]
    assert header == ['ファイル名', '総抽出ラベル数', '機器符号候補ラベル数', 'ページ数',
                      '検出領域数', '図番', 'タイトル', 'サブタイトル']
    row_a = [c.value for c in summary[2]]
    assert row_a == ['fileA.dxf', 10, 3, 1, 0, 'EE1234-500-01A', 'TITLE A', 'SUB A']

    ws = wb['抽出ラベル一覧']
    total_header = [c.value for c in ws[1]]
    assert total_header == ['機器符号候補', 'ラベル', '個数', '図番']
    total_rows = [(c[0].value, c[1].value, c[2].value) for c in ws.iter_rows(min_row=2)]
    assert total_rows == [('Y', 'CN1', 2), ('Y', 'R10', 1)]
    total_drawing_numbers = {row[1].value: row[3].value for row in ws.iter_rows(min_row=2)}
    assert total_drawing_numbers == {'CN1': 'EE1234-500-01A', 'R10': 'EE1234-500-01A'}


def test_create_ref_designator_excel_output_total_sheet_drawing_number_fallback_and_merge():
    """抽出ラベル一覧シートの図番列: 複数図番にまたがるラベルは ', ' 区切り・
    ソート済み、図番未抽出のファイルはファイル名（拡張子なし）にフォールバックする。"""
    results = {
        'a.dxf': {
            'rows': [{'機器符号候補': 'Y', 'ラベル': 'CN1', '個数': 1}], 'total_in_frame': 1,
            'ref_designator_count': 1, 'frames': 1,
            'main_drawing_number': 'EE5678-600-02B',
            'source_drawing_number': None, 'title': None, 'subtitle': None,
        },
        'b.dxf': {
            'rows': [{'機器符号候補': 'Y', 'ラベル': 'CN1', '個数': 1}], 'total_in_frame': 1,
            'ref_designator_count': 1, 'frames': 1,
            'main_drawing_number': 'EE1234-500-01A',
            'source_drawing_number': None, 'title': None, 'subtitle': None,
        },
        'c.dxf': {
            'rows': [{'機器符号候補': 'Y', 'ラベル': 'R10', '個数': 1}], 'total_in_frame': 1,
            'ref_designator_count': 1, 'frames': 1,
            'main_drawing_number': None,
            'source_drawing_number': None, 'title': None, 'subtitle': None,
        },
    }
    wb = _load_wb(create_ref_designator_excel_output(results, 'asc'))
    rows = {row[1].value: row[3].value for row in wb['抽出ラベル一覧'].iter_rows(min_row=2)}
    assert rows['CN1'] == 'EE1234-500-01A, EE5678-600-02B'
    assert rows['R10'] == 'c'   # 図番未抽出 → ファイル名（拡張子なし）


def test_create_ref_designator_excel_output_sort_desc():
    wb = _load_wb(create_ref_designator_excel_output(_make_ref_designator_results(), 'desc'))
    labels = [row[1].value for row in wb['fileA'].iter_rows(min_row=2)]
    assert labels == ['R10', 'CN1']


def test_create_ref_designator_excel_output_sorted_by_filename():
    """Summary の行順がファイル名昇順になる（投入順が逆でも並び替わる）。"""
    results = {
        'zzz.dxf': {'rows': [{'機器符号候補': 'Y', 'ラベル': 'R1', '個数': 1}], 'total_in_frame': 1,
                    'ref_designator_count': 1, 'frames': 1, 'main_drawing_number': None,
                    'source_drawing_number': None, 'title': None, 'subtitle': None},
        'aaa.dxf': {'rows': [{'機器符号候補': 'Y', 'ラベル': 'R2', '個数': 1}], 'total_in_frame': 1,
                    'ref_designator_count': 1, 'frames': 1, 'main_drawing_number': None,
                    'source_drawing_number': None, 'title': None, 'subtitle': None},
    }
    wb = _load_wb(create_ref_designator_excel_output(results, 'asc'))
    summary_filenames = [row[0].value for row in wb['Summary'].iter_rows(min_row=2)]
    assert summary_filenames == ['aaa.dxf', 'zzz.dxf']
    data_sheets = [n for n in wb.sheetnames if n not in ('Summary', '抽出ラベル一覧')]
    assert data_sheets == ['aaa', 'zzz']


def test_create_ref_designator_excel_output_marks_non_matching_labels_as_none():
    """機器符号候補パターンに一致しないラベル（スペース入り等）は「機器符号候補」
    列が None のまま出力される。"""
    results = {
        'a.dxf': {
            'rows': [
                {'機器符号候補': 'Y', 'ラベル': 'CN1', '個数': 1},
                {'機器符号候補': None, 'ラベル': 'NOTE ONE', '個数': 1},
            ],
            'total_in_frame': 2, 'ref_designator_count': 1, 'frames': 1,
            'main_drawing_number': None, 'source_drawing_number': None,
            'title': None, 'subtitle': None,
        },
    }
    wb = _load_wb(create_ref_designator_excel_output(results, 'asc'))
    ws = wb['a']
    rows = {row[1].value: row[0].value for row in ws.iter_rows(min_row=2)}
    assert rows['CN1'] == 'Y'
    assert rows['NOTE ONE'] is None


# ---------------------------------------------------------------------------
# create_region_excel_output
# ---------------------------------------------------------------------------

def _make_region_results(fname='circuit.dxf'):
    return {
        fname: {
            'rows': [
                {'機器符号候補': 'Y', 'ラベル': 'R10', '個数': 2, '領域': 'RACK1'},
                {'機器符号候補': 'Y', 'ラベル': 'CN3', '個数': 1, '領域': ''},
            ],
            'named': [
                {'polygon': [], 'name': 'RACK1', 'id': 0,
                 'frame': 0, 'area_pct': 25.0, 'label_count': 2},
            ],
            'frames': 1,
            'regions_detected': 1,
            'regions_named': 1,
            'total_in_frame': 3,
            'ref_designator_count': 3,
            'in_region_count': 2,
            'drawing_number': '',
            'title': None,
            'subtitle': None,
            'region_label_counts': {'RACK1': {'R10': 2}},
            'region_rows': [
                {'id': 0, 'frame': 0, 'name': 'RACK1', 'area_pct': 25.0,
                 'label_count': 2, 'ref_designator_count': 2},
            ],
        }
    }


def test_create_region_excel_output_returns_bytes():
    data = create_region_excel_output(_make_region_results())
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_create_region_excel_output_sheet_names():
    wb = _load_wb(create_region_excel_output(_make_region_results()))
    assert 'Summary' in wb.sheetnames
    assert '抽出ラベル一覧' in wb.sheetnames
    assert '領域一覧' in wb.sheetnames
    assert 'circuit' in wb.sheetnames


def test_create_region_excel_output_total_sheet_position_and_values():
    """`抽出ラベル一覧` シートは `Summary` 直後に配置され、全ファイル横断で
    ラベルをユニーク集計する（`rows` の『領域』列の値は無視して個数のみ合算）。"""
    wb = _load_wb(create_region_excel_output(_make_region_results()))
    assert wb.sheetnames.index('抽出ラベル一覧') == wb.sheetnames.index('Summary') + 1

    ws = wb['抽出ラベル一覧']
    header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    assert header == ['機器符号候補', 'ラベル', '個数', '図番']
    rows = {row[1].value: row[2].value for row in ws.iter_rows(min_row=2)}
    assert rows == {'R10': 2, 'CN3': 1}


def test_create_region_excel_output_total_sheet_drawing_number_merge():
    """複数ファイルにまたがる同一ラベルは、抽出ラベル一覧シートの図番列で
    ABC順・カンマ区切りに結合される（`drawing_number` 未抽出時はファイル名
    〔拡張子なし〕にフォールバック）。"""
    results = {}
    results.update(_make_region_results('a.dxf'))
    results.update(_make_region_results('b.dxf'))
    wb = _load_wb(create_region_excel_output(results))
    rows = {row[1].value: row[3].value for row in wb['抽出ラベル一覧'].iter_rows(min_row=2)}
    assert rows['R10'] == 'a, b'


def test_create_region_excel_output_total_sheet_absent_when_no_labels():
    """全ファイルにラベルが1件も無い場合、抽出ラベル一覧シート自体が作られない
    （他モードと同じ `skip_if_empty` 挙動）。"""
    results = _make_region_results()
    results['circuit.dxf']['rows'] = []
    wb = _load_wb(create_region_excel_output(results))
    assert '抽出ラベル一覧' not in wb.sheetnames


def test_create_region_excel_output_summary_row_count():
    """Summary シートの行数 = ヘッダー1行 + ファイル数"""
    results = _make_region_results()
    wb = _load_wb(create_region_excel_output(results))
    ws = wb['Summary']
    assert ws.max_row == 2   # ヘッダー + 1ファイル


def test_create_region_excel_output_empty():
    """ファイルなしでも例外なく返る"""
    data = create_region_excel_output({})
    wb = _load_wb(data)
    assert 'Summary' in wb.sheetnames


def test_create_region_excel_output_sorted_by_filename():
    """Summary の行順・ファイルシートの並びともファイル名昇順になる。"""
    results = {}
    results.update(_make_region_results('zzz.dxf'))
    results.update(_make_region_results('aaa.dxf'))
    wb = _load_wb(create_region_excel_output(results))
    summary_filenames = [row[0].value for row in wb['Summary'].iter_rows(min_row=2)]
    assert summary_filenames == ['aaa.dxf', 'zzz.dxf']
    data_sheets = [n for n in wb.sheetnames
                  if n not in ('Summary', '抽出ラベル一覧', '領域一覧', '領域別ラベル一覧')]
    assert data_sheets == ['aaa', 'zzz']


def test_create_region_excel_output_summary_always_has_eight_columns():
    """Summary シートは通常モード・領域モード共通の8列で、抽出結果の有無に
    かかわらず常にこの構成で出力される（v3.0.0、旧『図面番号・タイトル・
    サブタイトルを抽出』オプション廃止に伴い列の条件付き追加は廃止。値が
    無い場合は空文字を書き込むが、xlsxwriter/openpyxlの往復では空文字は
    Noneとして読み戻される既知の挙動のため、Noneで検証する）。"""
    wb = _load_wb(create_region_excel_output(_make_region_results()))
    header = [c.value for c in next(wb['Summary'].iter_rows(min_row=1, max_row=1))]
    assert header == ['ファイル名', '総抽出ラベル数', '機器符号候補ラベル数', 'ページ数',
                      '検出領域数', '図番', 'タイトル', 'サブタイトル']
    row = list(wb['Summary'].iter_rows(min_row=2, max_row=2, values_only=True))[0]
    row_dict = dict(zip(header, row))
    assert row_dict['図番'] is None
    assert row_dict['タイトル'] is None
    assert row_dict['サブタイトル'] is None


def test_create_region_excel_output_summary_includes_drawing_number_and_title_values():
    """図番・タイトル・サブタイトルが抽出されている場合はその値がそのまま入る
    （`DXF-label-compare` の『B側図番フィルタ』が要求する『図番』『タイトル』列を
    満たす。2026-07-23の欠落修正 v3.0.0でも維持）。"""
    results = _make_region_results()
    results['circuit.dxf']['drawing_number'] = 'EE0001-001-01A'
    results['circuit.dxf']['title'] = 'UNIT内結線図'
    results['circuit.dxf']['subtitle'] = 'SUB'
    wb = _load_wb(create_region_excel_output(results))
    header = [c.value for c in next(wb['Summary'].iter_rows(min_row=1, max_row=1))]
    row = list(wb['Summary'].iter_rows(min_row=2, max_row=2, values_only=True))[0]
    row_dict = dict(zip(header, row))
    assert row_dict['図番'] == 'EE0001-001-01A'
    assert row_dict['タイトル'] == 'UNIT内結線図'
    assert row_dict['サブタイトル'] == 'SUB'


# ---------------------------------------------------------------------------
# 半角正規化（出力ファイルのラベル・領域名はすべて半角で集計・記録する）
# ---------------------------------------------------------------------------

def test_normalize_width_basic():
    from model.common_utils import normalize_width
    assert normalize_width('ＳＹＳＴＥＭ　Ｉ／Ｆ　ＢＯＸ') == 'SYSTEM I/F BOX'
    assert normalize_width('ＣＮ１') == 'CN1'
    assert normalize_width('CN1') == 'CN1'
    assert normalize_width('ラック１') == 'ラック1'   # かなは不変・数字のみ半角化
    assert normalize_width('') == ''


def test_zenkaku_circuit_symbol_recognized_by_filter():
    """全角の機器符号（ＣＮ１）も filter_non_circuit_symbols が機器符号として
    認識する（判定は半角相当・返り値は元の表記のまま）。"""
    from model.common_utils import filter_non_circuit_symbols
    matched, excluded = filter_non_circuit_symbols(['ＣＮ１', 'これは説明文'])
    assert matched == ['ＣＮ１']
    assert excluded == 1


# ---------------------------------------------------------------------------
# 「領域一覧」刷新（v3.0.0: ページNo.列廃止・全領域記載・ファイル名×領域名でマージ）
# ---------------------------------------------------------------------------

def test_region_list_sheet_has_no_page_no_column():
    """v3.0.0で「ページ No.」列は廃止された。領域一覧のヘッダーは
    ファイル名・領域名・面積率[%]・領域内ラベル数・領域内機器符号候補数の5列。"""
    wb = _load_wb(create_region_excel_output(_make_region_results()))
    ws = wb['領域一覧']
    headers = [c.value for c in ws[1]]
    assert headers == ['ファイル名', '領域名', '面積率[%]', '領域内ラベル数', '領域内機器符号候補数']
    assert 'ページ No.' not in headers
    assert '図面' not in headers


def test_create_region_excel_output_region_list_merges_by_filename_and_name():
    """`region_rows` に同一ファイル・同一領域名の行が複数（ページ横断）ある場合、
    面積率・領域内ラベル数・領域内機器符号候補数を合算して1行にまとめる
    （v3.0.0、`ページ No.` 廃止に伴う仕様）。"""
    results = _make_region_results()
    results['circuit.dxf']['region_rows'] = [
        {'id': 0, 'frame': 0, 'name': 'RACK1', 'area_pct': 20.0, 'label_count': 3, 'ref_designator_count': 2},
        {'id': 1, 'frame': 1, 'name': 'RACK1', 'area_pct': 15.0, 'label_count': 1, 'ref_designator_count': 1},
        {'id': 2, 'frame': 0, 'name': 'CM DRV, PSW UNIT', 'area_pct': 10.0, 'label_count': 0, 'ref_designator_count': 0},
    ]
    wb = _load_wb(create_region_excel_output(results))
    ws = wb['領域一覧']
    rows = {row[1]: row for row in ws.iter_rows(min_row=2, values_only=True)}
    assert rows['RACK1'] == ('circuit.dxf', 'RACK1', 35.0, 4, 3)
    assert rows['CM DRV, PSW UNIT'] == ('circuit.dxf', 'CM DRV, PSW UNIT', 10.0, 0, 0)


def test_create_region_excel_output_region_list_autofilter_and_summary_no_filter():
    """Summary シート以外の全シートにオートフィルタが設定される（v3.0.0）。"""
    wb = _load_wb(create_region_excel_output(_make_region_results()))
    assert wb['Summary'].auto_filter.ref is None
    assert wb['領域一覧'].auto_filter.ref is not None
    assert wb['抽出ラベル一覧'].auto_filter.ref is not None
    assert wb['circuit'].auto_filter.ref is not None


def _make_multi_file_region_results():
    """2ファイルが同じ領域名 'RACK1' を共有し、ラベルが一部重なる合成データ。
    RACK1: R001 は両ファイルに登場（合計 8 = 5+3）、GND は file1 のみ（file2 は 0）。
    file2 には drawing_number が無い（未抽出）ためファイル名にフォールバックする。"""
    return {
        'a.dxf': {
            'rows': [], 'named': [
                {'polygon': [], 'name': 'RACK1', 'id': 0,
                 'frame': 0, 'area_pct': 30.0, 'label_count': 6},
            ],
            'frames': 1, 'regions_detected': 1, 'regions_named': 1,
            'total_in_frame': 6, 'ref_designator_count': 6,
            'in_region_count': 6,
            'drawing_number': 'EE6492-039-38A',
            'title': None, 'subtitle': None,
            'region_label_counts': {'RACK1': {'R001': 5, 'GND': 1}},
            'region_rows': [
                {'id': 0, 'frame': 0, 'name': 'RACK1', 'area_pct': 30.0,
                 'label_count': 6, 'ref_designator_count': 6},
            ],
        },
        'b.dxf': {
            'rows': [], 'named': [
                {'polygon': [], 'name': 'RACK1', 'id': 0,
                 'frame': 0, 'area_pct': 30.0, 'label_count': 3},
            ],
            'frames': 1, 'regions_detected': 1, 'regions_named': 1,
            'total_in_frame': 3, 'ref_designator_count': 3,
            'in_region_count': 3,
            'drawing_number': '',   # 未抽出 → ファイル名にフォールバック
            'title': None, 'subtitle': None,
            'region_label_counts': {'RACK1': {'R001': 3}},
            'region_rows': [
                {'id': 0, 'frame': 0, 'name': 'RACK1', 'area_pct': 30.0,
                 'label_count': 3, 'ref_designator_count': 3},
            ],
        },
    }


def test_build_region_label_summary_aggregates_across_files():
    from model.region_detector import build_region_label_summary
    results = _make_multi_file_region_results()
    files, rows = build_region_label_summary(results)

    assert files == [('a.dxf', 'EE6492-039-38A'), ('b.dxf', 'b')]

    by_label = {r['ラベル']: r for r in rows if r['領域名'] == 'RACK1'}
    assert by_label['R001']['合計個数'] == 8
    assert by_label['R001']['per_file'] == {'a.dxf': 5, 'b.dxf': 3}
    assert by_label['GND']['合計個数'] == 1
    assert by_label['GND']['per_file'].get('b.dxf', 0) == 0   # file2 に無いラベルは0扱い


def test_create_region_excel_output_label_summary_sheet():
    """「領域別ラベル一覧」シートが「領域一覧」の直後に作成され、指定のヘッダー・
    データ配置になっていること。「機器符号候補」列は「ラベル」列の直左に挿入される
    （v3.0.0）。ファイルに存在しないラベルの個数は 0 で埋まる。"""
    results = _make_multi_file_region_results()
    wb = _load_wb(create_region_excel_output(results))

    assert wb.sheetnames.index('領域別ラベル一覧') == wb.sheetnames.index('領域一覧') + 1

    ws = wb['領域別ラベル一覧']
    header = [c.value for c in ws[1]]
    assert header == ['領域名', '機器符号候補', 'ラベル', '合計個数', '図番', '個数', '図番', '個数']

    rows = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        rows[row[2]] = row  # ラベル（col2）をキー

    r001 = rows['R001']
    assert r001[0] == 'RACK1'
    assert r001[1] == 'Y'   # R001 は機器符号候補パターンに一致
    assert r001[3] == 8
    assert r001[4] == 'EE6492-039-38A' and r001[5] == 5
    assert r001[6] == 'b' and r001[7] == 3

    gnd = rows['GND']
    assert gnd[3] == 1
    assert gnd[4] == 'EE6492-039-38A' and gnd[5] == 1
    assert gnd[6] == 'b' and gnd[7] == 0   # file2 に無いラベルは 0
