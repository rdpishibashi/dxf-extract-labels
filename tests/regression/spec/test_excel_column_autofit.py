"""Excel出力の列幅を最長データに合わせる（2026-10-17 ユーザー要望）。

受入条件:
  - 各列は最長データ（全角=2字換算）+余白に合わせる。上限 MAX_COLUMN_WIDTH
  - カンマ区切りで図番を複数並べる列（抽出ラベル一覧・TB List の『図番』、結線図-組立図比較の
    『A 図番』）は、データ長に関わらず DRAWING_LIST_COLUMN_WIDTH の固定幅
  - 単一値の列（Summary の図番など）は通常どおり自動調整
"""
import os
import sys
from io import BytesIO

import openpyxl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from model.excel_output import (  # noqa: E402
    create_ref_designator_excel_output, display_width, autofit_width,
    MAX_COLUMN_WIDTH, DRAWING_LIST_COLUMN_WIDTH)


XLSXWRITER_WIDTH_OFFSET = 0.7109375   # xlsxwriter は指定幅に固定の余白分を足して保存する


def _width(ws, col_letter):
    return round(ws.column_dimensions[col_letter].width - XLSXWRITER_WIDTH_OFFSET, 3)


def test_display_width_counts_fullwidth_as_two():
    assert display_width('ABC') == 3
    assert display_width('端子台') == 6
    assert display_width('Ⅲ') == 2   # ambiguous も Excel 上は全角幅


def test_autofit_width_follows_longest_value_and_caps():
    assert autofit_width('個数', [1, 22]) == display_width('個数') + 2
    assert autofit_width('ラベル', ['A' * 20]) == 22
    assert autofit_width('ラベル', ['A' * 200]) == MAX_COLUMN_WIDTH
    assert autofit_width('ラベル', [None, float('nan')]) == display_width('ラベル') + 2


def test_excel_columns_fit_data_and_drawing_lists_are_fixed():
    long_label = 'SYSTEM I/F BOX WITH LONG NAME'
    results = {
        'EE0001-000-01A.dxf': {
            'rows': [{'機器符号候補': 'CAN', 'ラベル': long_label, '個数': 3},
                     {'機器符号候補': None, 'ラベル': 'X', '個数': 1}],
            'total_in_frame': 4, 'ref_designator_count': 3, 'frames': 1,
            'main_drawing_number': 'EE0001-000-01A', 'title': '短いタイトル', 'subtitle': ''},
    }
    wb = openpyxl.load_workbook(BytesIO(create_ref_designator_excel_output(results, 'asc')))

    sheet = wb['EE0001-000-01A']
    assert _width(sheet, 'B') == display_width(long_label) + 2        # ラベル: 最長データに合わせる
    assert _width(sheet, 'A') == display_width('機器符号候補') + 2    # 見出しが最長
    assert _width(sheet, 'C') == display_width('個数') + 2

    total = wb['抽出ラベル一覧']
    assert _width(total, 'D') == DRAWING_LIST_COLUMN_WIDTH            # 図番カンマ列は固定

    summary = wb['Summary']
    assert _width(summary, 'A') == display_width('EE0001-000-01A.dxf') + 2
    assert _width(summary, 'F') == display_width('EE0001-000-01A') + 2   # 単一図番は自動調整
