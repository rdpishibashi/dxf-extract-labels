"""プレビューの列幅・シート選択 selectbox 幅の純関数テスト（2026-10-17）。"""
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import app  # noqa: E402
from model.excel_output import DRAWING_LIST_COLUMN_WIDTH, MAX_COLUMN_WIDTH  # noqa: E402


def test_sheet_select_width_follows_longest_name_with_minimum():
    assert app.sheet_select_width(['Summary']) == 160            # 短い名前は下限
    long_names = ['Summary', 'X' * 31]
    assert app.sheet_select_width(long_names) == app._chars_to_px(31) + 36
    assert app.sheet_select_width(['日本語の領域名シート']) > app.sheet_select_width(['A' * 10])
    assert app.sheet_select_width([]) == 160


def test_preview_columns_fit_data_and_cap():
    df = pd.DataFrame({'ラベル': ['A' * 200, 'B'], '個数': [1, 22]})
    config, total = app.preview_column_config(df)
    assert config['ラベル']['width'] == app._chars_to_px(MAX_COLUMN_WIDTH)
    assert total == sum(c['width'] for c in config.values())
    assert config['個数']['width'] < config['ラベル']['width']


def test_drawing_list_column_is_fixed_only_when_it_has_commas():
    many = pd.DataFrame({'図番': ['EE1-1, EE2-2', 'EE3-3'], 'A 図番': ['EE1-1, EE2-2', '']})
    config, _ = app.preview_column_config(many)
    assert config['図番']['width'] == app._chars_to_px(DRAWING_LIST_COLUMN_WIDTH)
    assert config['A 図番']['width'] == app._chars_to_px(DRAWING_LIST_COLUMN_WIDTH)
    single = pd.DataFrame({'図番': ['EE1-1', 'EE3-3']})
    assert app.preview_column_config(single)[0]['図番']['width'] < app._chars_to_px(DRAWING_LIST_COLUMN_WIDTH)
