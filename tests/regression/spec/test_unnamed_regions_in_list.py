"""名称のない領域も「領域一覧」に出す（2026-10-10、ユーザー決定）の仕様テスト。

**受入条件**:
  - 名称候補が0件の領域（無名領域）の表示名は `<出力名>（<面積%整数>%）`（例 `no name 2（65%）`）。
    出力名は既存の規則（図面内の無名領域が1つなら `no name`、複数なら `no name N`）。
    ファイル名は含めない（見出しに表示されているため）。
  - 無名領域の「特定」は既定ON（従来から自動採用していたため）。明示的にOFFにすると採用しない。
  - 無名領域の「除外」は領域内ラベルを取り除く（以前は無名領域は除外対象外だった）。
  - 「領域一覧」に出さない呼び出し側（DXF-label-comparison）は、キーが無いので従来どおり自動採用。
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st  # noqa: E402

from model import ref_designator  # noqa: E402
from view import region_selection  # noqa: E402

SQ = [(0, 0), (10, 0), (10, 10), (0, 10)]
SQ2 = [(20, 0), (30, 0), (30, 10), (20, 10)]


def _reg(rid, cands, pct, poly=SQ):
    return {'id': rid, 'frame': 0, 'polygon': poly, 'area_pct': pct,
            'name_candidates': list(cands), 'default_name': cands[0][1] if cands else ''}


@pytest.fixture(autouse=True)
def fresh_session_state():
    original = st.session_state
    st.session_state = {}
    yield
    st.session_state = original


def test_labels_single_unnamed_has_no_number():
    a = {'regions': [_reg(0, [(1.0, 'RACK1')], 40), _reg(1, [], 65.4)]}
    assert ref_designator.unnamed_region_labels(a) == {1: 'no name（65%）'}


def test_labels_multiple_unnamed_are_numbered_in_region_order():
    a = {'regions': [_reg(0, [], 65.4), _reg(1, [(1.0, 'X')], 5), _reg(2, [], 6.0)]}
    assert ref_designator.unnamed_region_labels(a) == {0: 'no name 1（65%）', 2: 'no name 2（6%）'}


def test_label_has_no_filename():
    a = {'regions': [_reg(0, [], 30)]}
    assert '.dxf' not in ref_designator.unnamed_region_labels(a)[0]


def test_default_is_adopted_when_key_absent():
    """キーが無い（一覧に出さない呼び出し側）→ 従来どおり自動採用（選択辞書に何も入らない）。"""
    a = {'regions': [_reg(0, [], 30)]}
    assert region_selection.gather_name_selections('a.dxf', a) == {}


def test_explicit_off_means_not_adopted():
    a = {'regions': [_reg(0, [], 30)]}
    st.session_state['grc_no name（30%）'] = False
    sel = region_selection.gather_name_selections('a.dxf', a)
    assert sel == {('a.dxf', 0): ['']}
    named, _ = ref_designator.build_named_regions(a, sel, 'a.dxf')
    assert named == []


def test_on_is_adopted_with_existing_no_name():
    a = {'regions': [_reg(0, [], 30)]}
    st.session_state['grc_no name（30%）'] = True
    sel = region_selection.gather_name_selections('a.dxf', a)
    named, _ = ref_designator.build_named_regions(a, sel, 'a.dxf')
    assert [n['name'] for n in named] == ['no name']


def test_exclude_unnamed_region():
    a = {'regions': [_reg(0, [], 30), _reg(1, [(1.0, 'RACK1')], 40, SQ2)]}
    assert ref_designator.excluded_region_ids(a, {'no name（30%）'}) == {0}
    st.session_state['gre_no name（30%）'] = True
    assert region_selection.gather_name_selections('a.dxf', a) == {}  # 除外領域は名称を決めない


def test_numbering_stays_consistent_when_one_unnamed_is_off():
    a = {'regions': [_reg(0, [], 60), _reg(1, [], 6, SQ2)]}
    st.session_state['grc_no name 1（60%）'] = False
    sel = region_selection.gather_name_selections('a.dxf', a)
    named, _ = ref_designator.build_named_regions(a, sel, 'a.dxf')
    assert [n['name'] for n in named] == ['no name 2']


def test_summary_row_marks_off_unnamed_as_blank_and_keeps_name():
    a = {'regions': [_reg(0, [], 30)]}
    sel = {('a.dxf', 0): ['']}
    rows = ref_designator.build_all_regions_summary(a, sel, 'a.dxf', [], named_ids=frozenset())
    assert rows[0]['name'] == 'no name' and rows[0]['mark'] == ''
