"""不具合再発防止（2026-10-23 領域選択の状態）。

領域選択のレビューで見つかった、画面と出力の食い違い・選択のずれ。
  (1) 「領域の確認」で選んだ候補が、「抽出領域の選択」でほかの候補を外すと別の候補に変わった
      （選択状態 `rc_…_<i>` の添字が、チェック済み候補だけに絞った並びの番号だったため）
  (2) 「抽出領域の選択」で候補をすべて外した（特定OFF）領域に、以前入力した領域名が出力に残った
  (3) 「除外」した名称なし領域が、viewer では採用（赤）として表示された
  (4) 「図面を表示」OFF の件数表示が、伝播で立った `_none` を「選択しない」として数えた
  (5) 確定領域で入力名 "M1 DRV" → 「領域選択しない」ON → OFF にすると、入力欄に名称が残っているのに
      自動の名称（CN CS）に戻った（`editnone` が入力名の選択状態まで消していた）
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st  # noqa: E402

from model.ref_designator import unnamed_region_labels  # noqa: E402
from view import region_selection as rs  # noqa: E402

SQ = [(0, 0), (10, 0), (10, 10), (0, 10)]
LABELS = [('AAA', 1, 1), ('BBB', 2, 2), ('CCC', 3, 3)]


def _reg(rid, cands, poly=SQ):
    return {'id': rid, 'frame': 0, 'polygon': poly, 'corners': poly, 'area_pct': 50.0,
            'name_candidates': list(cands)}


@pytest.fixture
def state():
    original = st.session_state
    st.session_state = {}
    yield st.session_state
    st.session_state = original


def _init(reg, analysis):
    checked = rs.global_checked_region_names()
    st.session_state[rs.ANALYSES_KEY] = {'f': analysis}
    rs.ensure_choice_defaults('f', reg, rs.pending_candidates_for_region(reg, checked),
                              {'f': analysis}, checked)
    return checked


def _three_candidate_region(state):
    reg = _reg(0, [(1, 'AAA'), (2, 'BBB'), (3, 'CCC')])
    analysis = {'regions': [reg], 'labels': LABELS}
    for n in ('AAA', 'BBB', 'CCC'):
        state['grc_' + n] = True
    return reg, analysis


def test_chosen_candidate_survives_unchecking_another_candidate(state):
    """(1) BBB を選んだまま AAA を外しても、採用は BBB のまま。"""
    reg, analysis = _three_candidate_region(state)
    checked = _init(reg, analysis)
    rs.apply_viewer_choice('f', reg, 'cand', checked, idx=1, text='BBB')
    assert rs.gather_name_selections('f', analysis) == {('f', 0): ['BBB']}

    state['grc_AAA'] = False
    assert rs.gather_name_selections('f', analysis) == {('f', 0): ['BBB']}  # ensure の前でも
    _init(reg, analysis)  # 「領域の確認」の描画
    assert rs.gather_name_selections('f', analysis) == {('f', 0): ['BBB']}
    # 添字も新しい並び（BBB, CCC）に振り直されている
    assert state['rc_f_0_0'] is True and state['rc_f_0_1'] is False


def test_chosen_candidate_removed_falls_back_to_default(state):
    """(1') 選んでいた候補そのものを外すと、残った候補の既定（先頭）になる。"""
    reg, analysis = _three_candidate_region(state)
    checked = _init(reg, analysis)
    rs.apply_viewer_choice('f', reg, 'cand', checked, idx=2, text='CCC')
    state['grc_CCC'] = False
    _init(reg, analysis)
    assert rs.gather_name_selections('f', analysis) == {('f', 0): ['AAA']}


def test_none_choice_is_kept_when_candidates_change(state):
    """「領域選択しない」を選んでいれば、候補が変わっても実候補を勝手に選ばない。"""
    reg, analysis = _three_candidate_region(state)
    checked = _init(reg, analysis)
    rs.apply_viewer_choice('f', reg, 'none', checked)
    state['grc_AAA'] = False
    _init(reg, analysis)
    assert not any(state.get(f'rc_f_0_{i}') for i in range(3))
    assert state['rc_f_0_none'] is True


@pytest.mark.parametrize('override', ['typed', 'skip'])
def test_unspecified_region_ignores_typed_name_and_skip_mark(state, override):
    """(2) 特定OFF（チェック済み候補0件）の領域では、入力名・採用しない印が出力に残らない。"""
    reg = _reg(0, [(1, 'AAA')])
    analysis = {'regions': [reg], 'labels': LABELS}
    state['grc_AAA'] = False
    if override == 'typed':
        state['rc_f_0_typed'] = True
        state['rc_f_0_typedtext'] = 'ccc'
    else:
        state['rc_f_0_skip'] = True
    assert rs.gather_name_selections('f', analysis) == {}


def test_excluded_unnamed_region_is_inactive_in_viewer_not_adopted(state):
    """(3) 除外した名称なし領域は、viewer では「除外」の灰色（inactive）で、採用（unnamed）に入らない。"""
    import app
    analysis = {'regions': [_reg(0, [])], 'labels': [], 'main_drawing_number': 'X'}
    label = unnamed_region_labels(analysis)[0]
    state['grc_' + label] = True
    state['gre_' + label] = True
    assert app._unnamed_regions(analysis) == []
    assert [(r['id'], note) for r, _name, note in app._inactive_regions(analysis)] == [(0, '除外')]
    assert rs.gather_name_selections('f', analysis) == {}


def test_unnamed_region_stays_adopted_when_not_excluded(state):
    import app
    analysis = {'regions': [_reg(0, [])], 'labels': [], 'main_drawing_number': 'X'}
    got = app._unnamed_regions(analysis)
    assert [(r['id'], on) for r, _label, on in got] == [(0, True)]
    assert app._inactive_regions(analysis) == []


def test_excluded_unnamed_region_does_not_make_file_need_confirmation(state):
    """(3) 除外した名称なし領域だけを持つファイルは「選択の必要あり」にならない。"""
    analysis = {'regions': [_reg(0, [])], 'labels': [], 'main_drawing_number': 'X'}
    label = unnamed_region_labels(analysis)[0]
    state['gre_' + label] = True
    assert rs.region_confirmation_targets({'f': analysis}, set(), include_unnamed=True) == []
    state['gre_' + label] = False
    assert [t[0] for t in rs.region_confirmation_targets(
        {'f': analysis}, set(), include_unnamed=True)] == ['f']


@pytest.mark.parametrize('key,expected', [('rc_a.dxf_1_none', 0), ('rc_a.dxf_1_skip', 1)])
def test_viewer_off_summary_counts_skip_mark_not_propagated_none(key, expected):
    """(4) 「図面を表示」OFF の件数: 確定領域の採用しない印は `_skip`。伝播で立つ `_none` は数えない。"""
    from streamlit.testing.v1 import AppTest
    analysis = {'regions': [_reg(1, [(1.0, 'A')])], 'labels': [], 'frames': [1]}
    script = f'''
import sys
sys.path.insert(0, {PROJECT_ROOT!r})
import streamlit as st
from app import _render_region_confirmation_section
st.session_state['region_analyses'] = {{'a.dxf': {analysis!r}}}
st.session_state['region_selection_confirmed'] = True
st.session_state['grc_A'] = True
st.session_state[{key!r}] = True
class _UF:
    name = 'a.dxf'
    def getvalue(self): return b''
_render_region_confirmation_section(enable_region_detection=True, locked=False, uploaded_files=[_UF()])
'''
    at = AppTest.from_string(script, default_timeout=20).run()
    assert not at.exception
    assert any(f'名称を変更または選択しない {expected} 個' in c.value for c in at.caption)


def test_confirmed_region_returns_to_typed_name_after_none_is_cleared(state):
    """(5) 入力名 → 「領域選択しない」ON → OFF で、保持していた入力名に戻る。ON の間は採用せず灰色。"""
    import ezdxf, io
    from model import region_preview as rp
    reg = _reg(1, [(1.0, 'CN CS')])
    analysis = {'regions': [reg], 'labels': [('M1 DRV', 5, 5), ('CN CS', 6, 6)]}
    state['grc_CN CS'] = True
    checked = {'CN CS'}
    doc = ezdxf.new('R2010'); doc.modelspace().add_line((0, 0), (100, 100))
    buf = io.StringIO(); doc.write(buf)
    base = rp.render_base_svg(buf.getvalue().encode())

    def shown():
        data = rp.build_confirmation_data(
            base, 'f', analysis, [], [(reg, 'CN CS')], state, checked, False,
            rs.match_typed_region_name, lambda r, c: [])
        r = data['regions'][0]
        return r['state'], r['note'], r['name']

    rs.apply_viewer_choice('f', reg, 'typedtext', checked, typed_text='m1 drv')
    assert rs.gather_name_selections('f', analysis)[('f', 1)] == ['M1 DRV']
    assert shown() == ('decided', '', 'M1 DRV')

    rs.apply_viewer_choice('f', reg, 'editnone', checked)
    assert rs.gather_name_selections('f', analysis)[('f', 1)] == ['']
    assert shown()[:2] == ('unnamed_off', '領域選択しない')

    rs.apply_viewer_choice('f', reg, 'editnoneoff', checked)
    assert rs.gather_name_selections('f', analysis)[('f', 1)] == ['M1 DRV']
    assert shown() == ('decided', '', 'M1 DRV')
