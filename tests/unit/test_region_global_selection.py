"""「領域一覧」グローバル選択機能の単体テスト。

対象:
  - `model/region_detector.py`: `collect_unique_region_candidate_names()` /
    `resolve_globally_decided_name()` / `is_region_pending_selection()`
    （純粋関数、streamlit非依存）
  - `app.py`: `_gather_name_selections()`（グローバル決定分とファイル別UI
    選択分のマージ）。`st.session_state` を辞書に差し替えることで
    Streamlit ランタイム無しにテストする（`tests/unit/test_region_selection_sync.py`
    と同じ手法）。

検証済み仕様（2026-07-30 ユーザー指定）:
  - 候補0件の領域は対象外（自動 "no name" 命名は build_region_results 側）
  - 候補1件の領域は「領域一覧」チェックが唯一のゲート（未チェックなら抽出しない）
  - 候補2件以上の領域は、いずれかがチェックされていれば最優先の一致候補を採用、
    いずれもチェックされていなければ「領域の確認」での個別選択（従来通り）に委ねる
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model.region_detector import (  # noqa: E402
    collect_unique_region_candidate_names,
    resolve_globally_decided_name,
    is_region_pending_selection,
)

import streamlit as st  # noqa: E402
import app  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_session_state():
    st.session_state = {}
    yield
    st.session_state = {}


def _region(reg_id, candidates):
    return {'id': reg_id, 'name_candidates': list(candidates)}


# ---------------------------------------------------------------------------
# collect_unique_region_candidate_names
# ---------------------------------------------------------------------------

def test_collect_unique_names_dedupes_and_sorts():
    analyses = {
        'a.dxf': {'regions': [
            _region(0, [(1.0, 'RACK2'), (2.0, 'RACK1')]),
            _region(1, [(1.0, 'SYSTEM I/F BOX')]),
        ]},
        'b.dxf': {'regions': [
            _region(0, [(1.0, 'RACK1')]),  # 別ファイルの同名候補は重複除去
        ]},
    }
    result = collect_unique_region_candidate_names(analyses)
    assert result == ['RACK1', 'RACK2', 'SYSTEM I/F BOX']


def test_collect_unique_names_normalizes_width():
    """全角/半角の表記違いは同一名称としてユニーク化される。"""
    analyses = {
        'a.dxf': {'regions': [
            _region(0, [(1.0, 'ＲＡＣＫ１')]),
            _region(1, [(1.0, 'RACK1')]),
        ]},
    }
    result = collect_unique_region_candidate_names(analyses)
    assert result == ['RACK1']


def test_collect_unique_names_includes_single_candidate_regions():
    """候補1件のみの領域も一覧に含まれる（複数候補領域限定ではない）。"""
    analyses = {
        'a.dxf': {'regions': [_region(0, [(1.0, 'CTC')])]},
    }
    assert collect_unique_region_candidate_names(analyses) == ['CTC']


def test_collect_unique_names_empty_when_no_candidates():
    analyses = {'a.dxf': {'regions': [_region(0, [])]}}
    assert collect_unique_region_candidate_names(analyses) == []


# ---------------------------------------------------------------------------
# resolve_globally_decided_name
# ---------------------------------------------------------------------------

def test_resolve_globally_decided_name_picks_highest_priority_match():
    """候補リスト順（Tier→距離順）で最初に一致したものを採用する。"""
    reg = _region(0, [(1.0, 'RACK1'), (2.0, 'RACK2')])
    assert resolve_globally_decided_name(reg, {'RACK1', 'RACK2'}) == 'RACK1'
    assert resolve_globally_decided_name(reg, {'RACK2'}) == 'RACK2'


def test_resolve_globally_decided_name_none_when_no_match():
    reg = _region(0, [(1.0, 'RACK1')])
    assert resolve_globally_decided_name(reg, {'RACK2'}) is None
    assert resolve_globally_decided_name(reg, set()) is None


def test_resolve_globally_decided_name_normalizes_width():
    reg = _region(0, [(1.0, 'ＲＡＣＫ１')])
    assert resolve_globally_decided_name(reg, {'RACK1'}) == 'ＲＡＣＫ１'


# ---------------------------------------------------------------------------
# is_region_pending_selection
# ---------------------------------------------------------------------------

def test_is_pending_false_for_no_candidates():
    reg = _region(0, [])
    assert is_region_pending_selection(reg, set()) is False


def test_is_pending_false_for_single_candidate_regardless_of_check():
    """候補1件の領域は「領域一覧」チェックが唯一のゲートであり、
    「領域の確認」での個別選択対象にはならない（チェック有無を問わず False）。"""
    reg = _region(0, [(1.0, 'CTC')])
    assert is_region_pending_selection(reg, set()) is False
    assert is_region_pending_selection(reg, {'CTC'}) is False


def test_is_pending_true_for_multi_candidate_undecided():
    reg = _region(0, [(1.0, 'RACK1'), (2.0, 'RACK2')])
    assert is_region_pending_selection(reg, set()) is True
    assert is_region_pending_selection(reg, {'UNRELATED'}) is True


def test_is_pending_false_for_multi_candidate_decided():
    reg = _region(0, [(1.0, 'RACK1'), (2.0, 'RACK2')])
    assert is_region_pending_selection(reg, {'RACK2'}) is False


# ---------------------------------------------------------------------------
# app._gather_name_selections（グローバル決定＋ファイル別UIのマージ）
# ---------------------------------------------------------------------------

def test_gather_name_selections_uses_global_decision_for_multi_candidate():
    st.session_state['global_region_checked_names'] = ['RACK2']
    analysis = {'regions': [_region(5, [(1.0, 'RACK1'), (2.0, 'RACK2')])]}
    result = app._gather_name_selections('a.dxf', analysis)
    assert result == {('a.dxf', 5): ['RACK2']}


def test_gather_name_selections_single_candidate_requires_global_check():
    """候補1件の領域は「領域一覧」チェックが唯一のゲート。
    チェックが無ければ結果に含まれない（無所属）。"""
    analysis = {'regions': [_region(1, [(1.0, 'CTC')])]}

    st.session_state['global_region_checked_names'] = []
    assert app._gather_name_selections('a.dxf', analysis) == {}

    st.session_state['global_region_checked_names'] = ['CTC']
    assert app._gather_name_selections('a.dxf', analysis) == {('a.dxf', 1): ['CTC']}


def test_gather_name_selections_falls_back_to_checkbox_ui_when_undecided():
    """グローバル未決定（候補2件以上・いずれも未チェック）の領域は、
    「領域の確認」のチェックボックス状態（rc_<fname>_<reg_id>_<i>）から集める
    （従来通りの挙動）。"""
    st.session_state['global_region_checked_names'] = []
    st.session_state['rc_a.dxf_2_1'] = True  # 候補index1（'RACK2'）をチェック
    analysis = {'regions': [_region(2, [(1.0, 'RACK1'), (2.0, 'RACK2')])]}
    result = app._gather_name_selections('a.dxf', analysis)
    assert result == {('a.dxf', 2): ['RACK2']}


def test_gather_name_selections_no_candidates_region_excluded():
    """候補0件の領域はここでは何も返さない
    （build_region_results 等の自動 "no name" 命名に委ねる）。"""
    st.session_state['global_region_checked_names'] = []
    analysis = {'regions': [_region(3, [])]}
    assert app._gather_name_selections('a.dxf', analysis) == {}


def test_gather_name_selections_mixed_regions_in_one_file():
    """1ファイル内に「グローバル決定」「候補1件未チェック」「フォールバックUI」
    「候補0件」が混在していても、それぞれ正しく処理される。"""
    st.session_state['global_region_checked_names'] = ['SYSTEM I/F BOX']
    st.session_state['rc_a.dxf_10_0'] = True  # フォールバックUIでcandidates[0]選択
    analysis = {'regions': [
        _region(0, [(1.0, 'SYSTEM I/F BOX'), (2.0, 'OTHER')]),  # グローバル決定
        _region(1, [(1.0, 'CTC')]),                              # 候補1件・未チェック→除外
        _region(10, [(1.0, 'RACK1'), (2.0, 'RACK2')]),           # フォールバックUI
        _region(20, []),                                         # 候補0件
    ]}
    result = app._gather_name_selections('a.dxf', analysis)
    assert result == {
        ('a.dxf', 0): ['SYSTEM I/F BOX'],
        ('a.dxf', 10): ['RACK1'],
    }
