"""「領域の確認」: 同じファイルの別の図面で特定できた名称を優先する（2026-10-17 ユーザー指定）。

守るもの: 1ファイル複数図面で、ある図面の領域が複数候補（例 SYSTEM I/F BOX と CN-IF2-1）を
持っていても、別の図面でちょうど1件に決まった名称（SYSTEM I/F BOX）があれば、
その名称で決め、「領域の確認」に出さない。例: EE6892-039-03A（3図面）。

受入条件（Given/When/Then）:
  - Given 別の図面枠の領域が自動決定した名称がちょうど1種類、候補に一致 → その名称で決定・非表示
  - 一致が2種類以上 / 0種類 → 従来どおり要選択（表示）
  - 同じ図面枠内の別領域で決まった名称は根拠にしない
  - 別の図面枠の領域が「除外」指定されていれば根拠にしない
  - 図面が1つだけのファイルは影響を受けない
"""
import os
import sys

import pytest
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from view import region_selection as rs  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_session_state():
    original = st.session_state
    st.session_state = {}
    yield
    st.session_state = original


def _reg(reg_id, frame, *names):
    return {'id': reg_id, 'frame': frame, 'polygon': [], 'area_pct': 50.0,
            'name_candidates': [(float(i), n) for i, n in enumerate(names)]}


def _check(*names, exclude=()):
    for n in names:
        st.session_state[f'grc_{n}'] = True
    for n in exclude:
        st.session_state[f'gre_{n}'] = True


def _ee6892_like():
    """図面1だけ複数候補（SYSTEM I/F BOX・CN-IF2-1）、図面2・3は SYSTEM I/F BOX のみ。"""
    return {'regions': [
        _reg(0, 0, 'SYSTEM I/F BOX', 'CN-IF2-1'),
        _reg(1, 1, 'SYSTEM I/F BOX'),
        _reg(2, 2, 'SYSTEM I/F BOX'),
    ]}


def test_pending_region_resolved_by_name_decided_in_other_frames():
    _check('SYSTEM I/F BOX', 'CN-IF2-1')
    analysis = _ee6892_like()
    assert rs.gather_name_selections('a.dxf', analysis) == {
        ('a.dxf', 0): ['SYSTEM I/F BOX'],
        ('a.dxf', 1): ['SYSTEM I/F BOX'],
        ('a.dxf', 2): ['SYSTEM I/F BOX'],
    }
    assert rs.region_confirmation_targets({'a.dxf': analysis}, rs.global_checked_region_names()) == []


def test_still_pending_when_no_other_frame_decides_a_matching_name():
    _check('SYSTEM I/F BOX', 'CN-IF2-1')
    analysis = {'regions': [_reg(0, 0, 'SYSTEM I/F BOX', 'CN-IF2-1'), _reg(1, 1, 'GND BOX')]}
    targets = rs.region_confirmation_targets({'a.dxf': analysis}, rs.global_checked_region_names())
    assert [(f, [r['id'] for r in p]) for f, _a, p in targets] == [('a.dxf', [0])]


def test_still_pending_when_other_frames_decide_two_different_matching_names():
    _check('SYSTEM I/F BOX', 'CN-IF2-1')
    analysis = {'regions': [
        _reg(0, 0, 'SYSTEM I/F BOX', 'CN-IF2-1'),
        _reg(1, 1, 'SYSTEM I/F BOX'), _reg(2, 2, 'CN-IF2-1')]}
    targets = rs.region_confirmation_targets({'a.dxf': analysis}, rs.global_checked_region_names())
    assert [(f, [r['id'] for r in p]) for f, _a, p in targets] == [('a.dxf', [0])]


def test_name_decided_in_same_frame_is_not_a_basis():
    _check('SYSTEM I/F BOX', 'CN-IF2-1')
    analysis = {'regions': [_reg(0, 0, 'SYSTEM I/F BOX', 'CN-IF2-1'), _reg(1, 0, 'SYSTEM I/F BOX')]}
    targets = rs.region_confirmation_targets({'a.dxf': analysis}, rs.global_checked_region_names())
    assert [(f, [r['id'] for r in p]) for f, _a, p in targets] == [('a.dxf', [0])]


def test_excluded_region_in_other_frame_is_not_a_basis():
    _check('SYSTEM I/F BOX', 'CN-IF2-1')
    # 図面2の領域は候補 SYSTEM I/F BOX と DUMMY の2つ。DUMMY を除外指定すると領域全体が除外扱い
    analysis = {'regions': [_reg(0, 0, 'SYSTEM I/F BOX', 'CN-IF2-1'), _reg(1, 1, 'SYSTEM I/F BOX', 'DUMMY')]}
    st.session_state['gre_DUMMY'] = True
    targets = rs.region_confirmation_targets({'a.dxf': analysis}, rs.global_checked_region_names())
    assert [(f, [r['id'] for r in p]) for f, _a, p in targets] == [('a.dxf', [0])]


def test_single_frame_file_is_unaffected():
    _check('SYSTEM I/F BOX', 'CN-IF2-1')
    analysis = {'regions': [_reg(0, 0, 'SYSTEM I/F BOX', 'CN-IF2-1')]}
    targets = rs.region_confirmation_targets({'a.dxf': analysis}, rs.global_checked_region_names())
    assert [(f, [r['id'] for r in p]) for f, _a, p in targets] == [('a.dxf', [0])]
    assert rs.gather_name_selections('a.dxf', analysis) == {}
