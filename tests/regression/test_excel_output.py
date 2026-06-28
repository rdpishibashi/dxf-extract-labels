"""create_excel_output / create_region_excel_output のスモークテスト。

Excel 生成関数が最小の入力で例外なく完走し、
正しいシート構成の Excel を返すことを検証する。
ロジックの詳細（書式・列幅）ではなく「動作する・壊れていない」を確認する。
"""
import sys
import os
from io import BytesIO

import openpyxl
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from utils.excel_output import create_excel_output, create_region_excel_output  # noqa: E402


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _load_wb(data: bytes):
    return openpyxl.load_workbook(BytesIO(data))


# ---------------------------------------------------------------------------
# create_excel_output
# ---------------------------------------------------------------------------

def test_create_excel_output_returns_bytes():
    results = {
        '/tmp/foo.dxf': (['R10', 'CN3', 'R10'], {
            'filename': 'foo.dxf',
            'total_extracted': 3,
            'filtered_count': 0,
            'final_count': 3,
            'processed_layers': 1,
            'total_layers': 2,
            'main_drawing_number': 'EE0001-001-01A',
            'source_drawing_number': '',
            'title': 'TEST',
            'subtitle': '',
        }),
    }
    data = create_excel_output(results, False, 'asc', False)
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_create_excel_output_sheet_names():
    results = {
        '/tmp/circuit.dxf': (['R10', 'CN3'], {
            'filename': 'circuit.dxf',
            'total_extracted': 2,
            'filtered_count': 0,
            'final_count': 2,
            'processed_layers': 1,
            'total_layers': 1,
            'main_drawing_number': '',
            'source_drawing_number': '',
            'title': '',
            'subtitle': '',
        }),
    }
    wb = _load_wb(create_excel_output(results, False, 'asc', False))
    assert 'Summary' in wb.sheetnames
    assert 'Total' in wb.sheetnames
    assert 'circuit' in wb.sheetnames   # 拡張子なし


def test_create_excel_output_filter_mode():
    """filter_non_parts=True でも例外なく完走する"""
    results = {
        '/tmp/x.dxf': (['R10', 'CHAMBER', 'CN3'], {
            'filename': 'x.dxf',
            'total_extracted': 3,
            'filtered_count': 1,
            'final_count': 2,
            'processed_layers': 1,
            'total_layers': 1,
            'main_drawing_number': '',
            'source_drawing_number': '',
            'title': '',
            'subtitle': '',
        }),
    }
    data = create_excel_output(results, True, 'desc', False)
    wb = _load_wb(data)
    assert 'Summary' in wb.sheetnames


def test_create_excel_output_invalid_sheet():
    """validate_ref_designators=True かつ invalid ラベルがある場合 Invalid シートが作られる"""
    results = {
        '/tmp/y.dxf': (['R10', 'WEIRD'], {
            'filename': 'y.dxf',
            'total_extracted': 2,
            'filtered_count': 0,
            'final_count': 2,
            'processed_layers': 1,
            'total_layers': 1,
            'main_drawing_number': '',
            'source_drawing_number': '',
            'title': '',
            'subtitle': '',
            'invalid_ref_designators': ['WEIRD'],
        }),
    }
    wb = _load_wb(create_excel_output(results, False, 'asc', True))
    assert 'Invalid' in wb.sheetnames


def test_create_excel_output_empty_results():
    """ファイルが0件でも例外なく返る"""
    data = create_excel_output({}, False, 'asc', False)
    wb = _load_wb(data)
    assert 'Summary' in wb.sheetnames


# ---------------------------------------------------------------------------
# create_region_excel_output
# ---------------------------------------------------------------------------

def _make_region_results(fname='circuit.dxf'):
    return {
        fname: {
            'rows': [
                {'ラベル': 'R10', '個数': 2, '領域': 'RACK1'},
                {'ラベル': 'CN3', '個数': 1, '領域': ''},
            ],
            'named': [
                {'polygon': [], 'name': 'RACK1', 'id': 0,
                 'frame': 0, 'area_pct': 25.0, 'label_count': 2},
            ],
            'frames': 1,
            'regions_detected': 1,
            'regions_named': 1,
            'total_in_frame': 3,
            'filtered_count': 0,
            'final_count': 3,
            'in_region_count': 2,
        }
    }


def test_create_region_excel_output_returns_bytes():
    data = create_region_excel_output(_make_region_results())
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_create_region_excel_output_sheet_names():
    wb = _load_wb(create_region_excel_output(_make_region_results()))
    assert 'Summary' in wb.sheetnames
    assert '領域一覧' in wb.sheetnames
    assert 'circuit' in wb.sheetnames


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
