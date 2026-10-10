"""「領域一覧」表のチェック操作でスクロール位置が先頭に戻らないことを守る（2026-10-10）。

**以前の不具合**: チェックのたびにdata_editorのkeyのバージョンを上げて別ウィジェットとして
作り直していたため、表の表示位置が先頭行に戻り、続けてチェックする操作がしづらかった。
**保証すること**: keyのverはチェック操作では変わらず、外部から正本がクリアされた時
（「領域を検出」等）と候補名が変わった時だけ変わる。
"""
import os
import sys
from unittest import mock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from view.region_selection import region_list_editor_state  # noqa: E402

NAMES = ['A', 'B', 'C']
ALL_OFF = {n: (False, False) for n in NAMES}


def test_first_call_creates_state_with_base():
    s = region_list_editor_state(None, NAMES, ALL_OFF)
    assert s['ver'] == 0 and s['base'] == ALL_OFF and s['expected'] == ALL_OFF


def test_user_check_does_not_change_ver_or_base():
    """こちらがexpectedを更新して書き込んだ後の再実行ではverもbaseも不変。"""
    s0 = region_list_editor_state(None, NAMES, ALL_OFF)
    checked = dict(ALL_OFF, C=(True, False))
    s0['expected'] = checked  # render側がチェック操作を正本に反映した直後の状態
    s1 = region_list_editor_state(s0, NAMES, checked)
    assert s1['ver'] == 0 and s1['base'] == ALL_OFF


def test_external_clear_rebuilds():
    s0 = region_list_editor_state(None, NAMES, ALL_OFF)
    checked = dict(ALL_OFF, B=(True, True))
    s0['expected'] = checked
    s1 = region_list_editor_state(s0, NAMES, ALL_OFF)  # 正本が外部でクリアされた
    assert s1['ver'] == 1 and s1['base'] == ALL_OFF


def test_candidate_names_change_rebuilds():
    s0 = region_list_editor_state(None, NAMES, ALL_OFF)
    names2 = ['A', 'B', 'D']
    s1 = region_list_editor_state(s0, names2, {n: (False, False) for n in names2})
    assert s1['ver'] == 1


_SCRIPT = f'''
import sys
sys.path.insert(0, {PROJECT_ROOT!r})
from view.region_selection import render_global_region_names_section
render_global_region_names_section(['A', 'B'], locked=False, allow_exclude=True)
'''


def test_editor_args_column_order_width_and_hidden_index():
    """列順は 特定→除外→領域名、チェック列は64px、indexは非表示、領域名は編集不可。"""
    captured = {}
    real = st.data_editor

    def spy(df, **kw):
        captured['df'] = df
        captured['kw'] = kw
        return real(df, **kw)

    with mock.patch.object(st, 'data_editor', spy):
        at = AppTest.from_string(_SCRIPT)
        at.run()
    assert at.exception.len == 0
    kw = captured['kw']
    assert kw['column_order'] == ['特定', '除外', '領域名']
    assert kw['hide_index'] is True
    assert kw['disabled'] == ['領域名']
    assert kw['column_config']['特定']['width'] == 64
    assert kw['column_config']['除外']['width'] == 64
    assert list(captured['df'].columns) == ['特定', '除外', '領域名']
