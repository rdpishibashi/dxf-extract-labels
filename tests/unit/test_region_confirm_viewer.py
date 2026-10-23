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
from view import region_selection as rs  # noqa: E402

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
LABELS = [('rack1', 5, 12), ('RACK2', 5, -3), ('ＡＢ１', 50, 50), ('RACK1', 80, 80)]


def _reg(rid, cands, poly=SQUARE, frame=0):
    return {'id': rid, 'frame': frame, 'polygon': poly, 'corners': poly,
            'area_pct': 12.0, 'name_candidates': list(cands)}


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
    st.session_state['rc_f_1_typed'] = True

    st.session_state['rc_f_1_typedtext'] = 'rack2'
    assert rs.gather_name_selections('f', analysis) == {('f', 1): ['RACK2']}
    st.session_state['rc_f_1_typedtext'] = 'zzz'
    assert rs.gather_name_selections('f', analysis) == {('f', 1): ['']}  # 不一致＝未選択（採用しない印）
    st.session_state['rc_f_1_typedtext'] = 'ab1'
    assert rs.gather_name_selections('f', analysis) == {('f', 1): ['ＡＢ１']}
    st.session_state['rc_f_1_typedtext'] = ''
    assert rs.gather_name_selections('f', analysis) == {('f', 1): ['']}


def test_typed_is_exclusive_with_candidates_and_none(state):
    st.session_state['grc_A'] = st.session_state['grc_B'] = True
    st.session_state['region_analyses'] = {'f': {'regions': [_reg(1, [(1.0, 'A'), (2.0, 'B')])], 'labels': LABELS}}
    checked = {'A', 'B'}
    reg = st.session_state['region_analyses']['f']['regions'][0]
    st.session_state['rc_f_1_0'] = True

    rs.apply_viewer_choice('f', reg, 'typedtext', checked, typed_text='rack2')
    assert st.session_state['rc_f_1_typed'] is True
    assert not st.session_state['rc_f_1_0'] and not st.session_state['rc_f_1_1']
    assert not st.session_state.get('rc_f_1_none')

    rs.apply_viewer_choice('f', reg, 'cand', checked, idx=1, text='B')  # 候補を選ぶと typed は外れる
    assert st.session_state['rc_f_1_typed'] is False and st.session_state['rc_f_1_1'] is True

    rs.apply_viewer_choice('f', reg, 'none', checked)
    assert st.session_state['rc_f_1_none'] is True and st.session_state['rc_f_1_typed'] is False


def test_typed_region_is_not_touched_by_propagation(state):
    """他領域の候補選択の伝播は、「領域名を入力」を選んだ領域に及ばない。"""
    st.session_state['grc_X'] = st.session_state['grc_Y'] = True
    far = [(100, 0), (110, 0), (110, 10), (100, 10)]
    a, b = _reg(0, [(1.0, 'X'), (2.0, 'Y')]), _reg(1, [(1.0, 'X'), (2.0, 'Y')], poly=far)
    st.session_state['region_analyses'] = {'f': {'regions': [a, b], 'labels': LABELS}}
    st.session_state.update({'rc_f_1_typed': True, 'rc_f_1_0': False, 'rc_f_1_1': False})
    rs.apply_viewer_choice('f', a, 'cand', {'X', 'Y'}, idx=0, text='X')
    assert st.session_state['rc_f_1_0'] is False and st.session_state['rc_f_1_typed'] is True


def test_ensure_choice_defaults_is_idempotent(state):
    st.session_state['region_analyses'] = {}
    reg = _reg(3, [(1.0, 'A'), (2.0, 'B')])
    cands = [(1.0, 'A'), (2.0, 'B')]
    rs.ensure_choice_defaults('f', reg, cands, {}, {'A', 'B'})
    st.session_state['rc_f_3_1'] = True
    st.session_state['rc_f_3_0'] = False
    rs.ensure_choice_defaults('f', reg, cands, {}, {'A', 'B'})  # 既存の選択は上書きしない
    assert st.session_state['rc_f_3_1'] is True and st.session_state['rc_f_3_0'] is False
    assert st.session_state['rc_f_3_typed'] is False and st.session_state['rc_f_3_typedtext'] == ''


# ---------------------------------------------------------------------------
# viewer ON/OFF をまたいでも選択が消えない（AppTest）
# ---------------------------------------------------------------------------

def _viewer_toggle_app():
    import streamlit as st
    from view import region_selection as rs
    st.session_state.setdefault('grc_A', True)
    st.session_state.setdefault('grc_B', True)
    poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
    reg = {'id': 1, 'frame': 0, 'polygon': poly, 'corners': poly, 'area_pct': 10.0,
           'name_candidates': [(1.0, 'A'), (2.0, 'B')]}
    analyses = {'f.dxf': {'regions': [reg], 'frames': [1], 'labels': [('RACK2', 1, 1)]}}
    st.session_state['region_analyses'] = analyses

    viewer_on = st.toggle("viewer", key="viewer_on")
    rs.render_region_confirmation_section(
        analyses, False, file_preview=lambda f, a, p: viewer_on, allow_typed_name=True)
    st.write("rc=" + repr(st.session_state.get('rc_f.dxf_1_1')))


def test_selection_survives_viewer_on_off():
    at = AppTest.from_function(_viewer_toggle_app, default_timeout=20).run()
    assert not at.exception
    # viewer OFF: ページ上のチェックボックス（rcw_…）で2番目の候補を選ぶ
    boxes = {c.key: c for c in at.checkbox}
    boxes['rcw_f.dxf_1_1'].check().run()
    assert at.session_state['rc_f.dxf_1_1'] is True and at.session_state['rc_f.dxf_1_0'] is False

    at.toggle(key='viewer_on').set_value(True).run()  # viewer ON: チェックボックスは描かれない
    assert 'rcw_f.dxf_1_1' not in {c.key for c in at.checkbox}
    assert at.session_state['rc_f.dxf_1_1'] is True

    at.toggle(key='viewer_on').set_value(False).run()  # viewer OFF に戻す: 選択が反映される
    assert at.checkbox(key='rcw_f.dxf_1_1').value is True
    assert at.checkbox(key='rcw_f.dxf_1_0').value is False


def test_typed_input_ui_can_be_retried_and_shows_result():
    at = AppTest.from_function(_viewer_toggle_app, default_timeout=20).run()
    at.text_input(key='rcwt_f.dxf_1').set_value('rack2').run()
    assert at.session_state['rc_f.dxf_1_typed'] is True
    assert any('採用: RACK2' in c.value for c in at.caption)
    at.text_input(key='rcwt_f.dxf_1').set_value('nope').run()  # 入力し直す
    assert any('一致するラベルなし' in c.value for c in at.caption)
    assert at.session_state['rc_f.dxf_1_typedtext'] == 'nope'


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


def test_preview_overlays_put_name_on_every_original_label():
    analysis = {'regions': [_reg(0, [(1.0, 'A')])],
                'labels': [('A', 5, 12), ('a', 70, 12), ('B', 9, 9)]}
    rows = [{'id': 0, 'name': 'A', 'area_pct': 10, 'mark': '特定'}]
    ov = rp.preview_overlays(analysis, rows)[0]
    assert [l[1:] for l in ov.label_xys] == [(5, 12), (70, 12)]
    # 名称が複数候補（' | '）や選択済み複数（', '）でも、最初に見つかった名称のラベルへ
    rows = [{'id': 0, 'name': 'ZZZ | B', 'area_pct': 10, 'mark': ''}]
    assert [l[0] for l in rp.preview_overlays(analysis, rows)[0].label_xys] == ['B']
    # 該当ラベルが無い（no name 等）→ 空（多角形の左上に出す）
    rows = [{'id': 0, 'name': 'no name 1', 'area_pct': 10, 'mark': '特定'}]
    assert rp.preview_overlays(analysis, rows)[0].label_xys == []


def test_build_confirmation_data_states_and_selection(state):
    import ezdxf, io
    doc = ezdxf.new('R2010'); doc.modelspace().add_line((0, 0), (100, 100))
    buf = io.StringIO(); doc.write(buf)
    base = rp.render_base_svg(buf.getvalue().encode())
    pending = _reg(1, [(1.0, 'RACK1'), (2.0, 'RACK2')])
    decided = [(_reg(2, [(1.0, 'DONE')], poly=[(50, 50), (60, 50), (60, 60), (50, 60)]), 'DONE')]
    st.session_state.update({'rc_f_1_0': False, 'rc_f_1_1': True, 'rc_f_1_none': False,
                             'rc_f_1_typed': False, 'rc_f_1_typedtext': ''})
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
    import app
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

    app._REGION_CONFIRM_VIEWER = fake_viewer
    app._show_region_confirm_viewer('f.dxf', buf.getvalue().encode('utf-8'), analysis, [reg], False)
    st.write('mounted')


def test_viewer_call_passes_json_data_and_applies_returned_choice():
    at = AppTest.from_function(_viewer_mount_app, default_timeout=30).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.warning, [w.value for w in at.warning]
    assert at.session_state['_seen_regions'] == ['p1']
    assert at.session_state['rc_f.dxf_1_0'] is True  # 既定は先頭候補

    # viewer が候補2を選んだ（trigger 値）→ 正本が更新される。同じ seq の再送では再適用しない
    at.session_state['_fake_choice'] = {'seq': 1, 'region': 1, 'kind': 'cand', 'idx': 1, 'text': 'B'}
    at.run()
    assert at.session_state['rc_f.dxf_1_1'] is True and at.session_state['rc_f.dxf_1_0'] is False
    at.session_state['rc_f.dxf_1_0'] = True; at.session_state['rc_f.dxf_1_1'] = False
    at.run()  # seq は同じ → 無視
    assert at.session_state['rc_f.dxf_1_0'] is True

    at.session_state['_fake_choice'] = {'seq': 2, 'region': 1, 'kind': 'typedtext', 'typed_text': 'a'}
    at.run()
    assert at.session_state['rc_f.dxf_1_typed'] is True
    assert at.session_state['rc_f.dxf_1_typedtext'] == 'a'


def test_confirmation_targets_include_unnamed_only_when_requested():
    """名称の無い領域だけを持つファイルは、include_unnamed=True のときだけ「領域の確認」の対象になる。"""
    analysis = {'regions': [_reg(0, [])], 'labels': [], 'frames': [1]}
    assert rs.region_confirmation_targets({'f': analysis}, set()) == []
    targets = rs.region_confirmation_targets({'f': analysis}, set(), include_unnamed=True)
    assert [t[0] for t in targets] == ['f'] and targets[0][2] == []


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

    st.session_state.update({'rc_f_1_typed': True, 'rc_f_1_typedtext': 'm1 drv'})
    assert rs.gather_name_selections('f', analysis)[('f', 1)] == ['M1 DRV']
    st.session_state['rc_f_1_typedtext'] = 'xxx'
    assert rs.gather_name_selections('f', analysis)[('f', 1)] == ['']  # 不一致は未選択

    st.session_state.update({'rc_f_2_typed': True, 'rc_f_2_typedtext': 'm1 drv'})
    assert rs.gather_name_selections('f', analysis)[('f', 2)] == ['M1 DRV']  # 無名領域にも名前を付けられる

    rs.apply_viewer_choice('f', decided, 'typedoff', {'CN CS'})  # 外すと元の自動決定に戻る
    assert rs.gather_name_selections('f', analysis)[('f', 1)] == ['CN CS']


def test_viewer_data_lists_only_confirmed_regions_as_editable(state):
    import ezdxf, io
    doc = ezdxf.new('R2010'); doc.modelspace().add_line((0, 0), (100, 100))
    buf = io.StringIO(); doc.write(buf)
    base = rp.render_base_svg(buf.getvalue().encode())
    far = [(50, 50), (60, 50), (60, 60), (50, 60)]
    dec, un = _reg(1, [(1.0, 'CN CS')]), _reg(2, [], poly=far)
    st.session_state.update({'rc_f_1_typed': True, 'rc_f_1_typedtext': 'm1 drv'})
    data = rp.build_confirmation_data(
        base, 'f', {'labels': [('M1 DRV', 5, 5), ('CN CS', 6, 6)]}, [], [(dec, 'CN CS')], st.session_state,
        {'CN CS'}, False, rs.match_typed_region_name, lambda r, c: [], unnamed=[(un, 'no name 2', True)])
    by_key = {r['key']: r for r in data['regions']}
    assert by_key['d1']['name'] == 'M1 DRV' and by_key['d1']['state'] == 'decided'  # 入力名が優先
    assert by_key['d1']['editable'] and by_key['u2']['editable']
    assert data['choices']['1']['typed'] is True and data['choices']['1']['typedMatch'] == 'M1 DRV'
    assert data['choices']['2']['typed'] is False
