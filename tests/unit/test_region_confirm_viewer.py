"""
このテストが守るもの: 「領域の確認」の viewer 一体化（Part C）の、streamlit に依存しない部分と
選択状態の正本（rc_…）の扱い。
  - 「領域名を入力」: NFKC・大文字化した完全一致のみ採用、何度でも入力し直せる、不一致は未選択
  - viewer ON/OFF をまたいでも選択状態が消えない（正本 rc_ と widget key rcw_ の分離）
  - viewer 用データ（pending/decided・候補の選択・ラベル位置）
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from model import region_preview as rp  # noqa: E402
from model import region_choices as rc  # noqa: E402
from view import region_confirm_viewer as rcv  # noqa: E402
from view import region_selection as rs  # noqa: E402

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
LABELS = [('rack1', 5, 12), ('RACK2', 5, -3), ('ＡＢ１', 50, 50), ('RACK1', 80, 80)]


def _reg(rid, cands, poly=SQUARE, frame=0):
    return {'id': rid, 'frame': frame, 'polygon': poly, 'corners': poly,
            'area_pct': 12.0, 'name_candidates': list(cands)}


class _AtState:
    """AppTest の `at.session_state`（`get` を持たない）を、選択状態の読み書き（`region_choices`）に渡せる形にする。"""

    def __init__(self, at):
        self._s = at.session_state

    def get(self, key, default=None):
        return self._s[key] if key in self._s else default

    def __setitem__(self, key, value):
        self._s[key] = value


@pytest.fixture
def state():
    original = st.session_state
    st.session_state = {}
    yield st.session_state
    st.session_state = original


@pytest.mark.parametrize("typed,expected", [
    ("RACK2", "RACK2"), ("rack2", "RACK2"), ("ｒａｃｋ２", "RACK2"), ("  RACK2 ", "RACK2"),
    ("ab1", "ＡＢ１"),  # 採用名は一致したラベルの元の文字列
    ("RACK", None), ("", None), ("   ", None), ("NOPE", None),
])
def test_match_typed_region_name(typed, expected):
    assert rs.match_typed_region_name(typed, LABELS) == expected


def test_typed_name_follows_last_input_and_mismatch_means_unselected(state):
    """一致→不一致→一致と入力し直すと、採用は最後の入力に追従する。不一致なら未選択。"""
    st.session_state['grc_A'] = st.session_state['grc_B'] = True
    reg = _reg(1, [(1.0, 'A'), (2.0, 'B')])
    analysis = {'regions': [reg], 'labels': LABELS}
    rc.set_value(st.session_state, 'f', 1, 'typed', True)

    rc.set_value(st.session_state, 'f', 1, 'typedtext', 'rack2')
    assert rs.gather_name_selections('f', analysis) == {('f', 1): ['RACK2']}
    rc.set_value(st.session_state, 'f', 1, 'typedtext', 'zzz')
    assert rs.gather_name_selections('f', analysis) == {('f', 1): ['']}  # 不一致＝未選択（採用しない印）
    rc.set_value(st.session_state, 'f', 1, 'typedtext', 'ab1')
    assert rs.gather_name_selections('f', analysis) == {('f', 1): ['ＡＢ１']}
    rc.set_value(st.session_state, 'f', 1, 'typedtext', '')
    assert rs.gather_name_selections('f', analysis) == {('f', 1): ['']}


def test_typed_is_exclusive_with_candidates_and_none(state):
    st.session_state['grc_A'] = st.session_state['grc_B'] = True
    st.session_state['region_analyses'] = {'f': {'regions': [_reg(1, [(1.0, 'A'), (2.0, 'B')])], 'labels': LABELS}}
    checked = {'A', 'B'}
    reg = st.session_state['region_analyses']['f']['regions'][0]
    rc.set_value(st.session_state, 'f', 1, 0, True)

    rcv.apply_viewer_choice('f', reg, 'typedtext', checked, typed_text='rack2')
    assert rc.get_value(st.session_state, 'f', 1, 'typed') is True
    assert not rc.get_value(st.session_state, 'f', 1, 0) and not rc.get_value(st.session_state, 'f', 1, 1)
    assert not rc.get_value(st.session_state, 'f', 1, 'none')

    rcv.apply_viewer_choice('f', reg, 'cand', checked, idx=1, text='B')  # 候補を選ぶと typed は外れる
    assert rc.get_value(st.session_state, 'f', 1, 'typed') is False and rc.get_value(st.session_state, 'f', 1, 1) is True

    rcv.apply_viewer_choice('f', reg, 'none', checked)
    assert rc.get_value(st.session_state, 'f', 1, 'none') is True and rc.get_value(st.session_state, 'f', 1, 'typed') is False


def test_typed_region_is_not_touched_by_propagation(state):
    """他領域の候補選択の伝播は、「領域名を入力」を選んだ領域に及ばない。"""
    st.session_state['grc_X'] = st.session_state['grc_Y'] = True
    far = [(100, 0), (110, 0), (110, 10), (100, 10)]
    a, b = _reg(0, [(1.0, 'X'), (2.0, 'Y')]), _reg(1, [(1.0, 'X'), (2.0, 'Y')], poly=far)
    st.session_state['region_analyses'] = {'f': {'regions': [a, b], 'labels': LABELS}}
    rc.set_value(st.session_state, 'f', 1, 'typed', True)
    rc.set_value(st.session_state, 'f', 1, 0, False)
    rc.set_value(st.session_state, 'f', 1, 1, False)
    rcv.apply_viewer_choice('f', a, 'cand', {'X', 'Y'}, idx=0, text='X')
    assert rc.get_value(st.session_state, 'f', 1, 0) is False and rc.get_value(st.session_state, 'f', 1, 'typed') is True


def test_ensure_choice_defaults_is_idempotent(state):
    st.session_state['region_analyses'] = {}
    reg = _reg(3, [(1.0, 'A'), (2.0, 'B')])
    cands = [(1.0, 'A'), (2.0, 'B')]
    rs.ensure_choice_defaults('f', reg, cands, {}, {'A', 'B'})
    rc.set_value(st.session_state, 'f', 3, 1, True)
    rc.set_value(st.session_state, 'f', 3, 0, False)
    rs.ensure_choice_defaults('f', reg, cands, {}, {'A', 'B'})  # 既存の選択は上書きしない
    assert rc.get_value(st.session_state, 'f', 3, 1) is True and rc.get_value(st.session_state, 'f', 3, 0) is False
    assert rc.get_value(st.session_state, 'f', 3, 'typed') is False and rc.get_value(st.session_state, 'f', 3, 'typedtext') == ''


# ---------------------------------------------------------------------------
# viewer ON/OFF をまたいでも選択が消えない（AppTest）
# ---------------------------------------------------------------------------

def _page_ui_app():
    import streamlit as st
    from model import region_choices as rc
    from view import region_selection as rs
    st.session_state.setdefault('grc_A', True)
    st.session_state.setdefault('grc_B', True)
    poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
    reg = {'id': 1, 'frame': 0, 'polygon': poly, 'corners': poly, 'area_pct': 10.0,
           'name_candidates': [(1.0, 'A'), (2.0, 'B')]}
    analyses = {'f.dxf': {'regions': [reg], 'frames': [1], 'labels': [('RACK2', 1, 1)]}}
    st.session_state['region_analyses'] = analyses

    rs.render_region_confirmation_section(analyses, False)
    st.write("rc=" + repr(rc.get_value(st.session_state, 'f.dxf', 1, 1)))


def test_page_checkbox_selection_is_kept_across_reruns():
    """共有のページ上の選択（DXF-label-comparison・HostPL-comparer の経路）: チェックボックス（rcw_…）で
    選んだ候補が選択状態（rc_…）に写り、再描画してもチェックボックスに反映される。"""
    at = AppTest.from_function(_page_ui_app, default_timeout=20).run()
    assert not at.exception
    boxes = {c.key: c for c in at.checkbox}
    boxes['rcw_f.dxf_1_1'].check().run()
    assert rc.get_value(_AtState(at), 'f.dxf', 1, 1) is True and rc.get_value(_AtState(at), 'f.dxf', 1, 0) is False

    at.run()  # 再描画: 選択状態からチェックボックスへ写される
    assert at.checkbox(key='rcw_f.dxf_1_1').value is True
    assert at.checkbox(key='rcw_f.dxf_1_0').value is False


# ---------------------------------------------------------------------------
# viewer 用データ
# ---------------------------------------------------------------------------

def test_resolve_labels_returns_all_same_text_labels():
    """同じ文字列が複数あるのは領域が複数あるということ。1件に絞らずすべて返す。"""
    labels = [('RACK1', 80, 80), ('rack1', 5, 12), ('OTHER', 5, 5), ('ＲＡＣＫ１', 1, 1)]
    assert [l[1:] for l in rp.resolve_labels('RACK1', labels)] == [(80, 80), (5, 12), (1, 1)]
    assert rp.resolve_labels('NONE', labels) == []


def test_label_geom_uses_original_anchor_size_and_position():
    import ezdxf, io
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (100, 0), (100, 50), (0, 50)], close=True)
    m = msp.add_mtext('SYSTEM BOX', dxfattribs={'insert': (20, 10), 'char_height': 2.5, 'attachment_point': 5})
    buf = io.StringIO(); doc.write(buf)
    base = rp.render_base_svg(buf.getvalue().encode())
    g = rp.label_geom(base, ('SYSTEM BOX', 20, 10))
    assert (g['anchor'], g['base']) == ('middle', 'central')  # 挿入点が中央寄せの基準
    assert g['size'] and g['size'] > 0
    x, y = rp._to_svg(base.matrix, 20, 10)
    assert (g['x'], g['y']) == (x, y)
    # 情報の無いラベルは中央に置く
    assert rp.label_geom(base, ('NOPE', 1, 1))['anchor'] == 'middle'


def test_build_confirmation_data_states_and_selection(state):
    import ezdxf, io
    doc = ezdxf.new('R2010'); doc.modelspace().add_line((0, 0), (100, 100))
    buf = io.StringIO(); doc.write(buf)
    base = rp.render_base_svg(buf.getvalue().encode())
    pending = _reg(1, [(1.0, 'RACK1'), (2.0, 'RACK2')])
    decided = [(_reg(2, [(1.0, 'DONE')], poly=[(50, 50), (60, 50), (60, 60), (50, 60)]), 'DONE')]
    rc.set_value(st.session_state, 'f', 1, 0, False)
    rc.set_value(st.session_state, 'f', 1, 1, True)
    rc.set_value(st.session_state, 'f', 1, 'none', False)
    rc.set_value(st.session_state, 'f', 1, 'typed', False)
    rc.set_value(st.session_state, 'f', 1, 'typedtext', '')
    cands_fn = lambda reg, checked: reg['name_candidates']
    data = rp.build_confirmation_data(
        base, 'f', {'labels': [('rack1', 5, 12), ('RACK2', 5, -3), ('DONE', 52, 62)]}, [pending], decided,
        st.session_state, {'RACK1', 'RACK2'}, False, rs.match_typed_region_name, cands_fn)
    states = {r['key']: r['state'] for r in data['regions']}
    assert states == {'d2': 'decided', 'p1': 'pending'}
    cands = data['choices']['1']['cands']
    assert [c['selected'] for c in cands] == [False, True]
    assert all(c['geoms'] for c in cands)  # 元のラベルの位置が付く
    assert data['colors']['decided'] == '#ff3030'
    import json; json.dumps(data)  # JSON 化できる


def _viewer_mount_app():
    """app.py の viewer 呼び出し（data の JSON 化・戻り値 choice の反映）を確かめる。

    AppTest は Runtime を MagicMock にするため v2 部品の登録は使えない。部品の mount 関数を
    差し替え、viewer が返す choice を模擬する（部品の JS 自体は実ブラウザで確認する）。"""
    import io
    import json
    import types
    import ezdxf
    import streamlit as st
    from model import region_choices as rc
    from view import region_confirm_viewer as rcv
    from view import region_selection as rs

    st.session_state.setdefault('grc_A', True)
    st.session_state.setdefault('grc_B', True)
    doc = ezdxf.new('R2010')
    doc.modelspace().add_lwpolyline([(0, 0), (100, 0), (100, 50), (0, 50)], close=True)
    buf = io.StringIO(); doc.write(buf)
    poly = [(10, 10), (40, 10), (40, 30), (10, 30)]
    reg = {'id': 1, 'frame': 0, 'polygon': poly, 'corners': poly, 'area_pct': 10.0,
           'name_candidates': [(1.0, 'A'), (2.0, 'B')]}
    analysis = {'regions': [reg], 'frames': [1], 'labels': [('A', 12, 12)]}
    st.session_state['region_analyses'] = {'f.dxf': analysis}
    rs.ensure_choice_defaults('f.dxf', reg, reg['name_candidates'], {}, {'A', 'B'})

    def fake_viewer(key, data, height, on_choice_change):
        json.dumps(data)  # JSON 化できること
        st.session_state['_seen_regions'] = [r['key'] for r in data['regions']]
        return types.SimpleNamespace(choice=st.session_state.get('_fake_choice'))

    rcv.mount_viewer = fake_viewer
    rcv.show_viewer('f.dxf', buf.getvalue().encode('utf-8'), analysis, [reg], False)
    st.write('mounted')


def test_viewer_call_passes_json_data_and_applies_returned_choice():
    at = AppTest.from_function(_viewer_mount_app, default_timeout=30).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.warning, [w.value for w in at.warning]
    assert at.session_state['_seen_regions'] == ['p1']
    assert rc.get_value(_AtState(at), 'f.dxf', 1, 0) is True  # 既定は先頭候補

    # viewer が候補2を選んだ（trigger 値）→ 正本が更新される。同じ seq の再送では再適用しない
    at.session_state['_fake_choice'] = {'seq': 1, 'region': 1, 'kind': 'cand', 'idx': 1, 'text': 'B'}
    at.run()
    assert rc.get_value(_AtState(at), 'f.dxf', 1, 1) is True and rc.get_value(_AtState(at), 'f.dxf', 1, 0) is False
    rc.set_value(_AtState(at), 'f.dxf', 1, 0, True); rc.set_value(_AtState(at), 'f.dxf', 1, 1, False)
    at.run()  # seq は同じ → 無視
    assert rc.get_value(_AtState(at), 'f.dxf', 1, 0) is True

    at.session_state['_fake_choice'] = {'seq': 2, 'region': 1, 'kind': 'typedtext', 'typed_text': 'a'}
    at.run()
    assert rc.get_value(_AtState(at), 'f.dxf', 1, 'typed') is True
    assert rc.get_value(_AtState(at), 'f.dxf', 1, 'typedtext') == 'a'


def test_unnamed_only_file_is_a_confirmation_target_only_in_the_viewer_version():
    """名称の無い領域だけを持つファイルは、ページ上の選択（共有）では「領域の確認」の対象にならず、
    viewer 版（DXF-extract-labels）では対象で、確認が必要なファイルとして扱われる。"""
    analysis = {'regions': [_reg(0, [])], 'labels': [], 'frames': [1]}
    assert rs.region_confirmation_targets({'f': analysis}, set()) == []
    targets = rcv.confirmation_targets({'f': analysis}, set())
    assert [t[0] for t in targets] == ['f'] and targets[0][2] == []
    assert rcv.needs_choice(analysis, [], set()) is True


def test_build_confirmation_data_includes_unnamed_regions(state):
    import ezdxf, io
    doc = ezdxf.new('R2010'); doc.modelspace().add_line((0, 0), (100, 100))
    buf = io.StringIO(); doc.write(buf)
    base = rp.render_base_svg(buf.getvalue().encode())
    un_on, un_off = _reg(5, []), _reg(6, [], poly=[(50, 50), (60, 50), (60, 60), (50, 60)])
    data = rp.build_confirmation_data(
        base, 'f', {'labels': []}, [], [], st.session_state, set(), False,
        rs.match_typed_region_name, lambda r, c: [],
        unnamed=[(un_on, 'no name 1', True), (un_off, 'no name 2', False)])
    assert {r['key']: r['state'] for r in data['regions']} == {'u5': 'unnamed', 'u6': 'unnamed_off'}


def test_typed_override_applies_to_decided_and_unnamed_regions(state):
    """確定済み（候補1件で自動決定）・無名の領域も、「領域名を入力」で名称を変更できる。
    一致しなければ未選択（['']＝採用しない印。自動決定の名称には戻らない）。"""
    st.session_state['grc_CN CS'] = True
    labels = [('M1 DRV', 5, 5), ('CN CS', 6, 6)]
    decided = _reg(1, [(1.0, 'CN CS')])
    unnamed = _reg(2, [])
    analysis = {'regions': [decided, unnamed], 'labels': labels}
    assert rs.gather_name_selections('f', analysis)[('f', 1)] == ['CN CS']  # 変更前

    rc.set_value(st.session_state, 'f', 1, 'typed', True)
    rc.set_value(st.session_state, 'f', 1, 'typedtext', 'm1 drv')
    assert rs.gather_name_selections('f', analysis)[('f', 1)] == ['M1 DRV']
    rc.set_value(st.session_state, 'f', 1, 'typedtext', 'xxx')
    assert rs.gather_name_selections('f', analysis)[('f', 1)] == ['']  # 不一致は未選択

    rc.set_value(st.session_state, 'f', 2, 'typed', True)
    rc.set_value(st.session_state, 'f', 2, 'typedtext', 'm1 drv')
    assert rs.gather_name_selections('f', analysis)[('f', 2)] == ['M1 DRV']  # 無名領域にも名前を付けられる

    rcv.apply_viewer_choice('f', decided, 'typedoff', {'CN CS'})  # 外すと元の自動決定に戻る
    assert rs.gather_name_selections('f', analysis)[('f', 1)] == ['CN CS']


def test_viewer_data_lists_only_confirmed_regions_as_editable(state):
    import ezdxf, io
    doc = ezdxf.new('R2010'); doc.modelspace().add_line((0, 0), (100, 100))
    buf = io.StringIO(); doc.write(buf)
    base = rp.render_base_svg(buf.getvalue().encode())
    far = [(50, 50), (60, 50), (60, 60), (50, 60)]
    dec, un = _reg(1, [(1.0, 'CN CS')]), _reg(2, [], poly=far)
    rc.set_value(st.session_state, 'f', 1, 'typed', True)
    rc.set_value(st.session_state, 'f', 1, 'typedtext', 'm1 drv')
    data = rp.build_confirmation_data(
        base, 'f', {'labels': [('M1 DRV', 5, 5), ('CN CS', 6, 6)]}, [], [(dec, 'CN CS')], st.session_state,
        {'CN CS'}, False, rs.match_typed_region_name, lambda r, c: [], unnamed=[(un, 'no name 2', True)])
    by_key = {r['key']: r for r in data['regions']}
    assert by_key['d1']['name'] == 'M1 DRV' and by_key['d1']['state'] == 'decided'  # 入力名が優先
    assert by_key['d1']['editable'] and by_key['u2']['editable']
    assert data['choices']['1']['typed'] is True and data['choices']['1']['typedMatch'] == 'M1 DRV'
    assert data['choices']['2']['typed'] is False


def test_viewer_version_targets_include_all_files_unlike_the_shared_page_version():
    """確定のみ・領域なしのファイルも、viewer 版（DXF-extract-labels）だけ「領域の確認」の対象になる
    （図面枠が見つからない既知の未対応ケースは常に対象外）。"""
    decided = {'regions': [_reg(0, [(1.0, 'A')])], 'labels': [], 'frames': [1]}
    empty = {'regions': [], 'labels': [], 'frames': [1]}
    no_frame = {'regions': [], 'error': rs.FRAME_NOT_FOUND_ERROR_MARKER, 'frames': []}
    analyses = {'a': decided, 'b': empty, 'c': no_frame}
    assert rs.region_confirmation_targets(analyses, {'A'}) == []
    assert [t[0] for t in rcv.confirmation_targets(analyses, {'A'})] == ['a', 'b']


def test_viewer_data_shows_inactive_regions_gray_and_not_editable(state):
    """「除外」・特定OFF の領域は灰色で表示し、名称は変更できない。入力補完用に枠内のラベル名を渡す。"""
    import ezdxf, io
    doc = ezdxf.new('R2010'); doc.modelspace().add_line((0, 0), (100, 100))
    buf = io.StringIO(); doc.write(buf)
    base = rp.render_base_svg(buf.getvalue().encode())
    far = [(50, 50), (60, 50), (60, 60), (50, 60)]
    ex, off = _reg(3, [(1.0, 'EXC')]), _reg(4, [(1.0, 'OFF')], poly=far)
    data = rp.build_confirmation_data(
        base, 'f', {'labels': [('M1 DRV', 5, 5), ('CN CS', 6, 6), ('CN CS', 9, 9)]}, [], [], st.session_state,
        set(), False, rs.match_typed_region_name, lambda r, c: [],
        inactive=[(ex, 'EXC', '除外'), (off, 'OFF', '特定OFF')])
    by_key = {r['key']: r for r in data['regions']}
    assert {k: (r['state'], r['note'], r['editable']) for k, r in by_key.items()} == {
        'x3': ('unnamed_off', '除外', False), 'x4': ('unnamed_off', '特定OFF', False)}
    assert data['choices'] == {}
    assert data['labelNames'] == ['CN CS', 'M1 DRV']
    assert 'data-sg' in rcv.CONFIRM_JS and "normalize('NFKC')" in rcv.CONFIRM_JS


def test_confirmed_region_none_means_not_adopted_and_is_exclusive_with_typed(state):
    """確定済み・無名の領域の「（領域選択しない）」は採用しない印（['']）。入力名は保持され、
    外すと入力名に戻る（入力欄を空にすると自動の名称）。同じ候補を持つ他の確定領域には伝播しない。"""
    st.session_state['grc_CN CS'] = True
    labels = [('M1 DRV', 5, 5), ('CN CS', 6, 6)]
    a, b = _reg(1, [(1.0, 'CN CS')]), _reg(2, [(1.0, 'CN CS')])
    analysis = {'regions': [a, b], 'labels': labels}
    checked = {'CN CS'}

    rcv.apply_viewer_choice('f', a, 'editnone', checked)
    sel = rs.gather_name_selections('f', analysis)
    assert sel[('f', 1)] == [''] and sel[('f', 2)] == ['CN CS']

    rcv.apply_viewer_choice('f', a, 'typedtext', checked, typed_text='m1 drv')  # 入力すると「選択しない」は外れる
    assert rc.get_value(st.session_state, 'f', 1, 'skip') is False
    assert rs.gather_name_selections('f', analysis)[('f', 1)] == ['M1 DRV']
    rcv.apply_viewer_choice('f', a, 'editnone', checked)  # 「選択しない」の間は入力名を採用しない
    assert rc.get_value(st.session_state, 'f', 1, 'typed') is True
    assert rs.gather_name_selections('f', analysis)[('f', 1)] == ['']

    rcv.apply_viewer_choice('f', a, 'editnoneoff', checked)  # 外すと保持していた入力名に戻る
    assert rs.gather_name_selections('f', analysis)[('f', 1)] == ['M1 DRV']
    rcv.apply_viewer_choice('f', a, 'typedtext', checked, typed_text='m1 drv')
    rcv.apply_viewer_choice('f', a, 'typedoff', checked)  # 入力欄を空にする＝自動の名称に戻る
    assert rc.get_value(st.session_state, 'f', 1, 'typedtext') == ''
    assert rs.gather_name_selections('f', analysis)[('f', 1)] == ['CN CS']


def test_viewer_data_marks_none_region_gray(state):
    import ezdxf, io
    doc = ezdxf.new('R2010'); doc.modelspace().add_line((0, 0), (100, 100))
    buf = io.StringIO(); doc.write(buf)
    base = rp.render_base_svg(buf.getvalue().encode())
    dec = _reg(1, [(1.0, 'CN CS')])
    rc.set_value(st.session_state, 'f', 1, 'skip', True)
    data = rp.build_confirmation_data(
        base, 'f', {'labels': [('CN CS', 6, 6)]}, [], [(dec, 'CN CS')], st.session_state, {'CN CS'}, False,
        rs.match_typed_region_name, lambda r, c: [])
    reg = data['regions'][0]
    assert (reg['state'], reg['note'], reg['name']) == ('unnamed_off', '領域選択しない', 'CN CS')
    assert data['choices']['1']['none'] is True


def test_app_confirmation_wraps_every_file_in_expander_with_status_suffix():
    """全ファイルをファイル名のABC順の開閉枠に入れ、「（確定のみ）」「（選択の必要あり）」を付ける。"""
    two = {'regions': [dict(_reg(0, [(1.0, 'A'), (2.0, 'B')]), name_candidates=[(1.0, 'A'), (2.0, 'B')],
        dangling_edges=[{'attachment': (1.0, 2.0), 'entities': [{'handle': 'AB', 'start': (0, 0), 'end': (1, 1)}]}])],
           'labels': [], 'frames': [1]}
    one = {'regions': [_reg(0, [(1.0, 'C')])], 'labels': [], 'frames': [1]}
    noname = {'regions': [_reg(0, [])], 'labels': [], 'frames': [1]}
    script = f'''
import sys
sys.path.insert(0, {PROJECT_ROOT!r})
import streamlit as st
from app import _render_region_confirmation_section
st.session_state['region_analyses'] = {{'b.dxf': {two!r}, 'a.dxf': {one!r}, 'c.dxf': {noname!r}}}
st.session_state['region_selection_confirmed'] = True
for n in ('A', 'B', 'C'):
    st.session_state['grc_' + n] = True
class _UF:
    def __init__(self, name): self.name = name
    def getvalue(self): return b''
_render_region_confirmation_section(enable_region_detection=True, locked=False,
                                    uploaded_files=[_UF('a.dxf'), _UF('b.dxf'), _UF('c.dxf')])
'''
    at = AppTest.from_string(script, default_timeout=20).run()
    assert at.exception.len == 0
    assert len(at.get('popover')) >= 2  # 📐 と ⚠️行き止まり枝（開閉枠の中なのでポップオーバー）
    assert [e.label for e in at.expander] == [
        'a.dxf（確定のみ）', 'b.dxf（選択の必要あり）', 'c.dxf（選択の必要あり）']  # 名称なし領域も選択の必要あり


def test_none_propagated_from_pending_region_does_not_skip_confirmed_regions(state):
    """不具合再発防止（2026-10-23）: 要選択の領域で「領域選択しない」を選ぶと、同じ候補を持つ確定済み領域の
    内部状態 `_none` が True になる（既存の伝播）。これを確定領域の「採用しない」と読み違えて、
    確定のみの図面で一部の領域だけが「領域選択しない」の灰色になった。確定領域の採用しない印は別キー（_skip）。"""
    st.session_state['grc_A'] = st.session_state['grc_B'] = True
    far = [(50, 50), (60, 50), (60, 60), (50, 60)]
    pending = _reg(0, [(1.0, 'A'), (2.0, 'B')])
    decided = _reg(1, [(1.0, 'A')], poly=far)
    analysis = {'regions': [pending, decided], 'labels': []}
    st.session_state[rs.ANALYSES_KEY] = {'f': analysis}
    checked = {'A', 'B'}
    rs.ensure_choice_defaults('f', pending, rs.pending_candidates_for_region(pending, checked),
                              {'f': analysis}, checked)
    rcv.apply_viewer_choice('f', pending, 'none', checked)
    assert rc.get_value(st.session_state, 'f', 1, 'none') is True  # 既存の伝播（内部状態）
    assert rs.gather_name_selections('f', analysis).get(('f', 1)) == ['A']  # 確定領域は採用されたまま
