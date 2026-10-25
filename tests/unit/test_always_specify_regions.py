"""
このテストが守るもの（v3.18.0）: 「抽出領域の選択」セクションの廃止（検出した領域名は常に「特定」・「除外」は廃止）と、
採用される小領域（`minor_regions`）の viewer 表示・「領域の確認」での上書き。
  - 画面: 表・「すべての領域名を特定にする」・「領域選択を完了」が無く、「領域の確認」が検出直後から出る
  - `app._specify_all_region_names()` が全名称（名称候補・名称なし領域の表示名）を特定にし、除外は作らない
  - `model/minor_selection.py`: 採用される小領域（他ファイルで採用された名称と一致するものだけ）と上書き
  - viewer: 採用小領域が確定領域として出る（ラベルに「小領域」）。小領域にしかない名称は出ない。連結名は両方を強調
  - ファイルAでの選択が、ファイルBの小領域の採用（画面と出力の両方）に反映される
  - R1: 採用された小領域が持つ名称を、同じ図面枠の要選択の領域の選択肢から外す（`resolve_all()`）。
    小領域を「（領域選択しない）」「入力名」にすると選択肢に戻る
"""
import io
import json
import os
import sys

import ezdxf
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from model import minor_selection as ms  # noqa: E402
from model import region_choices as rc  # noqa: E402
from model import region_preview as rp  # noqa: E402
from model.ref_designator import OR_DELIMITER  # noqa: E402
from view import region_confirm_viewer as rcv  # noqa: E402
from view import region_selection as rs  # noqa: E402

BIG = [(0, 0), (40, 0), (40, 40), (0, 40)]
SMALL = [(50, 0), (60, 0), (60, 10), (50, 10)]
SMALL2 = [(70, 0), (80, 0), (80, 10), (70, 10)]


def _reg(rid, cands, poly=BIG, frame=0, pct=12.0):
    return {'id': rid, 'frame': frame, 'polygon': poly, 'corners': poly, 'area_pct': pct,
            'name_candidates': list(cands)}


def _analyses():
    """a.dxf: 通常領域 SYSTEM（図面枠0）＋小領域 CN-IF2-1・ONLY-MINOR。b.dxf: 通常領域 CN-IF2-1。"""
    a = {'regions': [_reg(0, [(1.0, 'SYSTEM')])],
         'minor_regions': [_reg(1, [(1.0, 'CN-IF2-1')], SMALL, pct=2.0),
                           _reg(2, [(1.0, 'ONLY-MINOR')], SMALL2, pct=1.0)],
         'labels': [('CN-IF2-1', 55, 5), ('ONLY-MINOR', 75, 5), ('M1 DRV', 52, 8), ('SYSTEM', 5, 5)],
         'frames': [1]}
    b = {'regions': [_reg(0, [(1.0, 'CN-IF2-1')])], 'minor_regions': [],
         'labels': [('CN-IF2-1', 5, 5)], 'frames': [1]}
    return {'a.dxf': a, 'b.dxf': b}


@pytest.fixture
def state():
    original = st.session_state
    st.session_state = {}
    for n in ('SYSTEM', 'CN-IF2-1', 'ONLY-MINOR'):
        st.session_state['grc_' + n] = True
    yield st.session_state
    st.session_state = original


def _selections(analyses):
    return {f: rs.gather_name_selections(f, a) for f, a in analyses.items()}


# ---------------------------------------------------------------------------
# 採用される小領域
# ---------------------------------------------------------------------------

def test_adopted_minor_names_only_adopts_names_chosen_elsewhere(state):
    """他のファイルで採用された名称（CN-IF2-1）の小領域だけ採用する。小領域にしかない名称（ONLY-MINOR）は採用しない。"""
    analyses = _analyses()
    assert ms.adopted_minor_names(analyses, _selections(analyses)) == {'a.dxf': {1: 'CN-IF2-1'}}


def test_adopted_minor_names_keeps_or_joined_name(state):
    """候補が対象の複数名称に一致する小領域は、` | ` で連結した名前で採用する。"""
    analyses = _analyses()
    analyses['a.dxf']['minor_regions'][0]['name_candidates'] = [(1.0, 'CN-IF2-1'), (2.0, 'SYSTEM')]
    got = ms.adopted_minor_names(analyses, _selections(analyses))
    # 通常領域 SYSTEM と同じ図面枠なので SYSTEM は使わず、CN-IF2-1 だけで採用される
    assert got == {'a.dxf': {1: 'CN-IF2-1'}}
    analyses['b.dxf']['regions'][0]['frame'] = 1  # SYSTEM は図面枠0のまま → 同じ図面枠の採用済みではない
    analyses['a.dxf']['regions'][0]['frame'] = 1
    got = ms.adopted_minor_names(analyses, _selections(analyses))
    assert got['a.dxf'][1] == OR_DELIMITER.join(['CN-IF2-1', 'SYSTEM'])


def test_minor_name_selections_apply_viewer_overrides(state):
    """採用された小領域の名称選択に、「（領域選択しない）」「領域名を入力」を重ねる。"""
    analyses = _analyses()
    a = analyses['a.dxf']
    minor = {1: 'CN-IF2-1'}
    assert ms.minor_name_selections(st.session_state, 'a.dxf', a, minor) == {('a.dxf', 1): ['CN-IF2-1']}

    rc.set_value(st.session_state, 'a.dxf', 1, 'skip', True)  # 領域選択しない
    assert ms.minor_name_selections(st.session_state, 'a.dxf', a, minor) == {('a.dxf', 1): ['']}

    rc.set_value(st.session_state, 'a.dxf', 1, 'skip', False)
    rc.set_value(st.session_state, 'a.dxf', 1, 'typed', True)
    rc.set_value(st.session_state, 'a.dxf', 1, 'typedtext', 'm1 drv')  # 入力名が一致
    assert ms.minor_name_selections(st.session_state, 'a.dxf', a, minor) == {('a.dxf', 1): ['M1 DRV']}
    rc.set_value(st.session_state, 'a.dxf', 1, 'typedtext', 'nothing')  # 不一致は未選択
    assert ms.minor_name_selections(st.session_state, 'a.dxf', a, minor) == {('a.dxf', 1): ['']}
    assert ms.minor_name_selections(st.session_state, 'a.dxf', a, {}) == {}  # 採用されていなければ何も返さない


def test_selection_in_one_file_changes_minor_adoption_in_another(state):
    """ファイルAで要選択の候補を選ぶと、ファイルBの小領域の採用が変わる（画面〈viewer 用の一覧〉と出力が同じ関数で決まる）。"""
    st.session_state['grc_X'] = st.session_state['grc_Y'] = True
    a = {'regions': [_reg(0, [(1.0, 'X'), (2.0, 'Y')])], 'minor_regions': [], 'labels': [], 'frames': [1]}
    b = {'regions': [], 'minor_regions': [_reg(5, [(1.0, 'Y')], SMALL, pct=2.0)], 'labels': [('Y', 55, 5)],
         'frames': [1]}
    analyses = {'a.dxf': a, 'b.dxf': b}
    st.session_state['region_analyses'] = analyses
    rs.ensure_choice_defaults('a.dxf', a['regions'][0], a['regions'][0]['name_candidates'], analyses, {'X', 'Y'})
    assert rcv.minor_regions_for_viewer(analyses, 'b.dxf') == []  # 既定は先頭候補 X → Y の小領域は採用しない

    rcv.apply_viewer_choice('a.dxf', a['regions'][0], 'cand', {'X', 'Y'}, idx=1, text='Y')
    assert [(r['id'], n) for r, n in rcv.minor_regions_for_viewer(analyses, 'b.dxf')] == [(5, 'Y')]
    assert ms.adopted_minor_names(analyses, _selections(analyses)) == {'b.dxf': {5: 'Y'}}


# ---------------------------------------------------------------------------
# viewer の data
# ---------------------------------------------------------------------------

def _base():
    doc = ezdxf.new('R2010')
    doc.modelspace().add_line((0, 0), (100, 100))
    buf = io.StringIO()
    doc.write(buf)
    return rp.render_base_svg(buf.getvalue().encode())


def test_viewer_data_labels_minor_regions_and_highlights_every_name_of_an_or_name(state):
    analyses = _analyses()
    a = analyses['a.dxf']
    a['minor_regions'][0]['name_candidates'] = [(1.0, 'CN-IF2-1'), (2.0, 'M1 DRV')]
    minor = a['minor_regions'][0]
    name = OR_DELIMITER.join(['CN-IF2-1', 'M1 DRV'])
    data = rp.build_confirmation_data(
        _base(), 'a.dxf', a, [], [(minor, name)], st.session_state, set(), False,
        rs.match_typed_region_name, lambda r, c: [], minor_ids={1})
    json.dumps(data)
    reg = data['regions'][0]
    assert reg['state'] == 'decided' and reg['label'].endswith('2%・小領域）')
    assert sorted(g['text'] for g in reg['geoms']) == ['CN-IF2-1', 'M1 DRV']  # 連結名の両方のラベルを強調


def test_viewer_data_does_not_label_ordinary_regions_as_minor(state):
    a = _analyses()['a.dxf']
    data = rp.build_confirmation_data(
        _base(), 'a.dxf', a, [], [(a['regions'][0], 'SYSTEM')], st.session_state, set(), False,
        rs.match_typed_region_name, lambda r, c: [], minor_ids=set())
    assert '小領域' not in data['regions'][0]['label']


def _viewer_app():
    """show_viewer に analyses を渡したとき、採用小領域だけが確定領域として viewer の data に入る。"""
    import io
    import json
    import types
    import ezdxf
    import streamlit as st
    from view import region_confirm_viewer as rcv

    for n in ('SYSTEM', 'CN-IF2-1', 'ONLY-MINOR'):
        st.session_state.setdefault('grc_' + n, True)
    doc = ezdxf.new('R2010')
    doc.modelspace().add_lwpolyline([(0, 0), (100, 0), (100, 50), (0, 50)], close=True)
    buf = io.StringIO(); doc.write(buf)
    poly = lambda x0: [(x0, 0), (x0 + 10, 0), (x0 + 10, 10), (x0, 10)]  # noqa: E731
    reg = lambda rid, c, x0, pct: {'id': rid, 'frame': 0, 'polygon': poly(x0), 'corners': poly(x0),  # noqa: E731
                                   'area_pct': pct, 'name_candidates': [(1.0, c)]}
    a = {'regions': [reg(0, 'SYSTEM', 0, 12.0)],
         'minor_regions': [reg(1, 'CN-IF2-1', 50, 2.0), reg(2, 'ONLY-MINOR', 70, 1.0)],
         'labels': [('CN-IF2-1', 55, 5), ('ONLY-MINOR', 75, 5), ('SYSTEM', 5, 5)], 'frames': [1]}
    b = {'regions': [reg(0, 'CN-IF2-1', 0, 12.0)], 'minor_regions': [], 'labels': [('CN-IF2-1', 5, 5)],
         'frames': [1]}
    analyses = {'a.dxf': a, 'b.dxf': b}
    st.session_state['region_analyses'] = analyses

    def fake_viewer(key, data, height, on_choice_change):
        json.dumps(data)
        st.session_state['_seen'] = {r['key']: r['label'] for r in data['regions']}
        return types.SimpleNamespace(choice=st.session_state.get('_fake_choice'))

    rcv.mount_viewer = fake_viewer
    rcv.show_viewer('a.dxf', buf.getvalue().encode('utf-8'), a, [], False, analyses)
    st.write('mounted')


def test_show_viewer_lists_only_adopted_minor_regions_and_applies_edits():
    at = AppTest.from_function(_viewer_app, default_timeout=30).run()
    assert not at.exception, [e.value for e in at.exception]
    seen = at.session_state['_seen']
    assert sorted(seen) == ['d0', 'd1']  # ONLY-MINOR（小領域にしかない名称）は出ない
    assert '小領域' in seen['d1'] and '小領域' not in seen['d0']

    # viewer が小領域に「（領域選択しない）」を返す → 正本に skip が立ち、出力の名称選択が採用しない印になる
    at.session_state['_fake_choice'] = {'seq': 1, 'region': 1, 'kind': 'editnone'}
    at.run()
    assert not at.exception, [e.value for e in at.exception]

    class _S:
        def __init__(self, s): self._s = s
        def get(self, k, default=None): return self._s[k] if k in self._s else default  # noqa: E704
    assert rc.get_value(_S(at.session_state), 'a.dxf', 1, 'skip') is True


# ---------------------------------------------------------------------------
# 画面
# ---------------------------------------------------------------------------

def test_specify_all_region_names_turns_on_every_name_and_never_creates_exclusion():
    analyses = {'a.dxf': {'regions': [_reg(0, [(1.0, 'A'), (2.0, 'B')]), _reg(1, [])], 'labels': []}}
    script = f'''
import sys
sys.path.insert(0, {PROJECT_ROOT!r})
import streamlit as st
from app import _specify_all_region_names
analyses = {analyses!r}
_specify_all_region_names(analyses)
st.session_state['_keys'] = sorted(k for k in st.session_state.to_dict() if k.startswith(('grc_', 'gre_')))
'''
    at = AppTest.from_string(script, default_timeout=20).run()
    assert at.exception.len == 0
    keys = at.session_state['_keys']
    assert not any(k.startswith('gre_') for k in keys)
    assert {'grc_A', 'grc_B'} <= set(keys) and len(keys) == 3  # 名称なし領域の表示名も1つ
    assert all(at.session_state[k] is True for k in keys)


def test_app_no_longer_has_the_region_selection_table_or_complete_button():
    """「領域の確認」は「領域選択を完了」なしで検出直後から出る。表・一括「特定」・「領域選択を完了」は無い。"""
    import app
    assert not hasattr(app, '_render_global_region_names_section')
    assert not hasattr(rcv, 'render_select_all_button')
    one = {'regions': [_reg(0, [(1.0, 'C')])], 'labels': [], 'frames': [1]}
    script = f'''
import sys
sys.path.insert(0, {PROJECT_ROOT!r})
import streamlit as st
from app import _render_region_confirmation_section, _specify_all_region_names
analyses = {{'a.dxf': {one!r}}}
st.session_state['region_analyses'] = analyses
_specify_all_region_names(analyses)
class _UF:
    def __init__(self, name): self.name = name
    def getvalue(self): return b''
_render_region_confirmation_section(enable_region_detection=True, locked=False, uploaded_files=[_UF('a.dxf')])
'''
    at = AppTest.from_string(script, default_timeout=20).run()
    assert at.exception.len == 0
    assert [e.label for e in at.expander] == ['a.dxf（確定のみ）']  # 「領域選択を完了」を押していなくても出る
    assert not [b for b in at.button if b.label in ('領域選択を完了', 'すべての領域名を特定にする')]
    assert not at.get('arrow_data_frame') and not [sh for sh in at.subheader if sh.value == '抽出領域の選択']


# ---------------------------------------------------------------------------
# R1: 採用された小領域が持つ名称を、要選択の領域の選択肢から外す（EE3273-039-06B を模す）
# ---------------------------------------------------------------------------

def _r1_analyses(cands=('A', 'B'), minors=('B',), with_other_file=True):
    """a.dxf: 要選択の通常領域 id0（候補 cands）＋小領域（minors。id は 2 から）。b.dxf: 通常領域 B（B を採用済みにする）。"""
    regs = [_reg(0, [(1.0 + i, c) for i, c in enumerate(cands)])]
    mins = [_reg(2 + i, [(1.0, m)], [(50 + 20 * i, 0), (60 + 20 * i, 0), (60 + 20 * i, 10), (50 + 20 * i, 10)], pct=2.0)
            for i, m in enumerate(minors)]
    a = {'regions': regs, 'minor_regions': mins, 'frames': [1],
         'labels': [('M1 DRV', 52, 8), ('A', 5, 5), ('B', 55, 5), ('C', 6, 6)]}
    out = {'a.dxf': a}
    if with_other_file:
        out['b.dxf'] = {'regions': [_reg(0, [(1.0, 'B')])], 'minor_regions': [], 'labels': [('B', 5, 5)], 'frames': [1]}
    return out


@pytest.fixture
def r1_state():
    original = st.session_state
    st.session_state = {}
    for n in ('A', 'B', 'C'):
        st.session_state['grc_' + n] = True
    yield st.session_state
    st.session_state = original


def _resolve(analyses):
    return ms.resolve_all(st.session_state, analyses, rs.classify_file_regions)


def test_r1_pending_region_is_decided_when_an_adopted_minor_holds_one_candidate(r1_state):
    """要望そのもの: 小領域（B）が採用されていれば、同じ図面枠の要選択の領域（A,B）は A で確定する。"""
    analyses = _r1_analyses()
    res = _resolve(analyses)['a.dxf']
    assert (res.decisions[0].status, res.decisions[0].name) == ('decided', 'A')
    assert res.selections[('a.dxf', 0)] == ['A'] and res.selections[('a.dxf', 2)] == ['B']
    assert res.hidden == {0: {'B'}}
    assert [(r['id'], n) for r, n in res.minors] == [(2, 'B')]
    # 画面（要選択の一覧）からも外れる
    checked = rs.global_checked_region_names()
    pend = {f: p for f, _a, p in rcv.confirmation_targets(analyses, checked, _resolve(analyses))}
    assert pend['a.dxf'] == []
    assert [r['id'] for r in {f: p for f, _a, p in rcv.confirmation_targets(analyses, checked)}['a.dxf']] == [0]


def test_r1_skip_or_typed_minor_returns_the_candidate_to_the_pending_region(r1_state):
    """小領域を「（領域選択しない）」にする、または入力名で別のラベルにすると、領域の選択肢に B が戻る。"""
    analyses = _r1_analyses()
    rc.set_value(st.session_state, 'a.dxf', 2, 'skip', True)
    res = _resolve(analyses)['a.dxf']
    assert res.decisions[0].status == 'pending' and [t for _d, t in res.decisions[0].candidates] == ['A', 'B']
    assert res.hidden == {} and res.selections[('a.dxf', 2)] == [''] and [r['id'] for r, _n in res.minors] == [2]

    rc.set_value(st.session_state, 'a.dxf', 2, 'skip', False)
    rc.set_value(st.session_state, 'a.dxf', 2, 'typed', True)
    rc.set_value(st.session_state, 'a.dxf', 2, 'typedtext', 'm1 drv')
    res = _resolve(analyses)['a.dxf']
    assert res.decisions[0].status == 'pending' and res.selections[('a.dxf', 2)] == ['M1 DRV']  # 入力名が持つのは B ではない


def test_r1_three_candidates_keep_original_indexes_in_the_viewer(r1_state):
    """候補が3件で B だけ外れる → 要選択（A,C）。viewer の idx は元の番号（0,2）で、選ぶと C が採用される。"""
    analyses = _r1_analyses(cands=('A', 'B', 'C'))
    a = analyses['a.dxf']
    reg0 = a['regions'][0]
    rs.ensure_choice_defaults('a.dxf', reg0, reg0['name_candidates'], analyses, {'A', 'B', 'C'})
    st.session_state['region_analyses'] = analyses
    res = _resolve(analyses)['a.dxf']
    assert res.decisions[0].status == 'pending' and [t for _d, t in res.decisions[0].candidates] == ['A', 'C']
    data = rp.build_confirmation_data(
        _base(), 'a.dxf', a, [reg0], [], st.session_state, {'A', 'B', 'C'}, False,
        rs.match_typed_region_name, lambda r, c: r['name_candidates'], hidden_cands=res.hidden)
    assert [(c['idx'], c['text']) for c in data['choices']['0']['cands']] == [(0, 'A'), (2, 'C')]

    rcv.apply_viewer_choice('a.dxf', reg0, 'cand', {'A', 'B', 'C'}, idx=2, text='C')
    assert _resolve(analyses)['a.dxf'].selections[('a.dxf', 0)] == ['C']


def test_r1_choice_in_the_pending_region_does_not_block_the_minor_adoption(r1_state):
    """循環を断つ: 要選択の領域で B を選んでいても、小領域 B は採用され、領域は A で確定する（B を使うには小領域を外す）。"""
    analyses = _r1_analyses(with_other_file=False)
    reg0 = analyses['a.dxf']['regions'][0]
    rs.ensure_choice_defaults('a.dxf', reg0, reg0['name_candidates'], analyses, {'A', 'B'})
    st.session_state['region_analyses'] = analyses
    rcv.apply_viewer_choice('a.dxf', reg0, 'cand', {'A', 'B'}, idx=1, text='B')
    res = _resolve(analyses)['a.dxf']
    assert [(r['id'], n) for r, n in res.minors] == [(2, 'B')]
    assert res.decisions[0].status == 'decided' and res.selections[('a.dxf', 0)] == ['A']


def test_r1_when_every_candidate_is_held_the_region_stays_pending_and_chosen_name_leaves_the_minor(r1_state):
    """全候補を小領域が持つときは要選択のまま。選んだ名称は、同じ図面枠の（上書きのない）小領域から外れる。"""
    analyses = _r1_analyses(minors=('A', 'B'))
    analyses['c.dxf'] = {'regions': [_reg(0, [(1.0, 'A')])], 'minor_regions': [], 'labels': [('A', 5, 5)], 'frames': [1]}  # A も採用済みにする
    reg0 = analyses['a.dxf']['regions'][0]
    rs.ensure_choice_defaults('a.dxf', reg0, reg0['name_candidates'], analyses, {'A', 'B'})
    st.session_state['region_analyses'] = analyses
    rcv.apply_viewer_choice('a.dxf', reg0, 'cand', {'A', 'B'}, idx=1, text='B')
    res = _resolve(analyses)['a.dxf']
    assert res.decisions[0].status == 'pending' and res.hidden == {}
    assert res.selections[('a.dxf', 0)] == ['B']
    assert ('a.dxf', 3) not in res.selections  # B を持つ小領域（id3）は採用されない
    assert [(r['id'], n) for r, n in res.minors] == [(2, 'A')]


def test_r1_resolve_all_is_pure_and_deterministic(r1_state):
    import copy
    analyses = _r1_analyses(cands=('A', 'B', 'C'))
    reg0 = analyses['a.dxf']['regions'][0]
    rs.ensure_choice_defaults('a.dxf', reg0, reg0['name_candidates'], analyses, {'A', 'B', 'C'})
    before = copy.deepcopy(st.session_state)
    first, second = _resolve(analyses), _resolve(analyses)
    assert st.session_state == before  # 状態を書き換えない
    assert {f: r.selections for f, r in first.items()} == {f: r.selections for f, r in second.items()}


def test_r1_extraction_uses_the_same_selections_as_the_viewer(r1_state):
    """画面と出力が食い違わない: 抽出は `region_confirm_viewer.resolve()` の selections をそのまま使う。"""
    analyses = _r1_analyses()
    st.session_state['region_analyses'] = analyses
    res = rcv.resolve(analyses)
    assert res['a.dxf'].selections == _resolve(analyses)['a.dxf'].selections
    assert rcv.minor_regions_for_viewer(analyses, 'a.dxf') == res['a.dxf'].minors
