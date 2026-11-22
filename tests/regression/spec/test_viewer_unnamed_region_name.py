"""仕様: 「領域の確認」の viewer は、名称のない領域を `no name`／`no name N` だけで表示する。

**受入条件**（2026-11-22、ユーザー要望。「図面を表示して領域を確認」の図面上の表示と左のリストに、`no name 1（2%）図番` と
出ていた）:

- Given 名称のない領域（候補0件）、When viewer に渡す表示データ（`region_preview.build_confirmation_data()`）を作る、
  Then その領域の `name` は出力名（`unnamed_output_names()`。無名領域が1つなら `no name`、複数なら `no name N`）だけで、
  面積（`（2%）`）も図番も含まない。面積は見出し `label`（`図面N / 領域M（P%）`）に出ている。
  「特定OFF」「除外」で灰色（`unnamed_off`）になった無名領域も同じ。
- 「特定」「除外」のキー（`unnamed_region_labels()`＝`no name N（P%）図番`）と Excel の領域名は変わらない
  （図面ごとに「特定」「除外」を区別するため、キーに図番が要る）。

境界・例外の意図: 表示名だけを変え、キーは共有の `ref_designator` のまま。`unnamed_regions()`・`inactive_regions()` が返す名前を使う。
"""
import io
import os
import sys

import ezdxf
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st  # noqa: E402

from model import ref_designator, region_preview as rp  # noqa: E402
from model.region_detector import pending_candidates_for_region  # noqa: E402
from view import region_confirm_viewer as rcv  # noqa: E402
from view import region_selection as rs  # noqa: E402

DRAWING = 'DE3527-553-10E'
SQ1 = [(0, 0), (10, 0), (10, 10), (0, 10)]
SQ2 = [(50, 50), (60, 50), (60, 60), (50, 60)]
SQ3 = [(70, 70), (80, 70), (80, 80), (70, 80)]


def _reg(rid, poly, pct):
    return {'id': rid, 'frame': 0, 'polygon': poly, 'corners': poly, 'area_pct': pct, 'name_candidates': []}


@pytest.fixture
def state():
    original = st.session_state
    st.session_state = {}
    yield st.session_state
    st.session_state = original


def _base():
    doc = ezdxf.new('R2010')
    doc.modelspace().add_line((0, 0), (100, 100))
    buf = io.StringIO()
    doc.write(buf)
    return rp.render_base_svg(buf.getvalue().encode())


def _data(analysis, base):
    decisions = rs.classify_file_regions(analysis)
    return rp.build_confirmation_data(
        base, 'f', analysis, [], [], st.session_state, set(), False,
        rs.match_typed_region_name, pending_candidates_for_region,
        unnamed=rcv.unnamed_regions(analysis, decisions), inactive=rcv.inactive_regions(analysis, decisions))


def _names(data):
    return {r['id']: (r['state'], r['name'], r['label']) for r in data['regions']}


def test_a_single_unnamed_region_is_shown_as_no_name_only(state):
    analysis = {'regions': [_reg(0, SQ1, 2.4)], 'labels': [], 'main_drawing_number': DRAWING}
    assert ref_designator.unnamed_region_labels(analysis)[0] == f'no name（2%）{DRAWING}'     # キーは変えない
    state[f'grc_no name（2%）{DRAWING}'] = True
    assert _names(_data(analysis, _base())) == {0: ('unnamed', 'no name', '図面1 / 領域1（2%）')}


def test_several_unnamed_regions_are_numbered_without_area_or_drawing_number(state):
    analysis = {'regions': [_reg(0, SQ1, 30.0), _reg(2, SQ2, 8.0)], 'labels': [], 'main_drawing_number': DRAWING}
    for label in ref_designator.unnamed_region_labels(analysis).values():
        state['grc_' + label] = True
    names = _names(_data(analysis, _base()))
    assert [(rid, name) for rid, (_s, name, _l) in sorted(names.items())] == [(0, 'no name 1'), (2, 'no name 2')]
    assert all('%' not in name and DRAWING not in name for _s, name, _l in names.values())
    assert names[2][2] == '図面1 / 領域3（8%）'                      # 面積は見出しに出ている


def test_unnamed_regions_turned_off_or_excluded_are_shown_the_same_way(state):
    """「特定OFF」（`grc_` が False）と「除外」（`gre_` が True）の無名領域は灰色（`unnamed_off`）で、名前は同じ形。"""
    analysis = {'regions': [_reg(0, SQ1, 30.0), _reg(1, SQ2, 8.0), _reg(2, SQ3, 5.0)],
                'labels': [], 'main_drawing_number': DRAWING}
    labels = ref_designator.unnamed_region_labels(analysis)
    state['grc_' + labels[0]] = True
    state['grc_' + labels[1]] = False                                # 特定OFF
    state['grc_' + labels[2]] = True
    state['gre_' + labels[2]] = True                                 # 除外
    names = _names(_data(analysis, _base()))
    assert {rid: (s, name) for rid, (s, name, _l) in names.items()} == {
        0: ('unnamed', 'no name 1'), 1: ('unnamed_off', 'no name 2'), 2: ('unnamed_off', 'no name 3')}
