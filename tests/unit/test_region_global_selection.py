"""「領域一覧」グローバル選択機能の単体テスト。

対象:
  - `model/region_detector.py`: `collect_unique_region_candidate_names()` /
    `resolve_globally_decided_name()` / `is_region_pending_selection()` /
    `pending_candidates_for_region()`（純粋関数、streamlit非依存）
  - `app.py`: `_gather_name_selections()`（グローバル決定分とファイル別UI
    選択分のマージ）。`st.session_state` を辞書に差し替えることで
    Streamlit ランタイム無しにテストする（`tests/unit/test_region_selection_sync.py`
    と同じ手法）。

検証済み仕様（2026-07-30 ユーザー指定。「領域一覧」でチェックされなかった
候補は「採用しない」という明示的な意思表示とみなす方針に統一）:
  - 候補0件の領域は対象外（自動 "no name" 命名は
    `ref_designator.build_named_regions()`/`build_all_regions_summary()` 側）
  - 候補1件の領域は「領域一覧」チェックが唯一のゲート（未チェックなら抽出しない）
  - 候補のうちチェック済みがちょうど1件の領域は、その候補を自動採用する
  - 候補のうちチェック済みが2件以上の領域のみ「領域の確認」での個別選択対象になる
    （未チェックの候補は選択肢に出さない。`pending_candidates_for_region()` で
    チェック済みのものだけに絞り込む）
  - チェック済みが0件（候補はあるが誰もチェックしていない）の領域は
    「領域の確認」にも出さず、無所属のまま（除外）
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
    pending_candidates_for_region,
)

import streamlit as st  # noqa: E402
import app  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_session_state():
    # `st.session_state` を辞書に差し替えて Streamlit ランタイム無しにテストする
    # 手法（本ファイルの docstring 参照）。ただし単純に `st.session_state = {}` で
    # 終わらせると、本来モジュール属性として存在する `SessionStateProxy`
    # インスタンスがテスト後も空dictのまま残ってしまい、同一プロセス内で後から
    # 実行される `streamlit.testing.v1.AppTest` ベースのテストの session_state
    # 分離が壊れる（2026-09-10、`test_region_list_no_candidates_message.py`
    # 実装時にこの汚染により不可解な失敗を発見・修正——テスト実行順序に
    # 依存する形で他テストの session_state が漏れ込んでいた）。
    # 元の（本物の）オブジェクトを保存し、テスト後に必ず復元する。
    original_session_state = st.session_state
    st.session_state = {}
    yield
    st.session_state = original_session_state


def _region(reg_id, candidates):
    return {'id': reg_id, 'name_candidates': list(candidates)}


def _set_global_checked(*names):
    """「領域一覧」のチェックボックス状態（`grc_<name>`）をテスト用に設定する。"""
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith('grc_'):
            del st.session_state[k]
    for name in names:
        st.session_state[f'grc_{name}'] = True


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
# pending_candidates_for_region
# ---------------------------------------------------------------------------

def test_pending_candidates_keeps_only_checked_in_original_order():
    reg = _region(0, [(1.0, 'RACK1'), (2.0, 'RACK2'), (3.0, 'RACK3')])
    result = pending_candidates_for_region(reg, {'RACK3', 'RACK1'})
    assert result == [(1.0, 'RACK1'), (3.0, 'RACK3')]


def test_pending_candidates_empty_when_none_checked():
    reg = _region(0, [(1.0, 'RACK1'), (2.0, 'RACK2')])
    assert pending_candidates_for_region(reg, set()) == []


def test_pending_candidates_normalizes_width():
    reg = _region(0, [(1.0, 'ＲＡＣＫ１'), (2.0, 'RACK2')])
    assert pending_candidates_for_region(reg, {'RACK1'}) == [(1.0, 'ＲＡＣＫ１')]


# ---------------------------------------------------------------------------
# resolve_globally_decided_name
# ---------------------------------------------------------------------------

def test_resolve_globally_decided_name_when_exactly_one_match():
    reg = _region(0, [(1.0, 'RACK1'), (2.0, 'RACK2')])
    assert resolve_globally_decided_name(reg, {'RACK2'}) == 'RACK2'


def test_resolve_globally_decided_name_none_when_no_match():
    reg = _region(0, [(1.0, 'RACK1')])
    assert resolve_globally_decided_name(reg, {'RACK2'}) is None
    assert resolve_globally_decided_name(reg, set()) is None


def test_resolve_globally_decided_name_none_when_multiple_matches():
    """候補が2件以上チェックされている場合は自動採用せず None を返す
    （「領域の確認」での個別選択に委ねる。2026-07-30、旧仕様〈最優先候補を
    自動採用〉から変更）。"""
    reg = _region(0, [(1.0, 'RACK1'), (2.0, 'RACK2')])
    assert resolve_globally_decided_name(reg, {'RACK1', 'RACK2'}) is None


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


def test_is_pending_false_when_no_candidate_checked():
    """候補が2件以上あっても、いずれもチェックされていなければ「採用しない」
    という意思表示とみなし、個別選択対象にはしない（2026-07-30、旧仕様
    〈未決定として個別選択対象にする〉から変更）。"""
    reg = _region(0, [(1.0, 'RACK1'), (2.0, 'RACK2')])
    assert is_region_pending_selection(reg, set()) is False
    assert is_region_pending_selection(reg, {'UNRELATED'}) is False


def test_is_pending_true_when_two_or_more_candidates_checked():
    """候補のうちチェック済みが2件以上ある場合のみ、どちらを採用するか
    個別選択が必要になる。"""
    reg = _region(0, [(1.0, 'RACK1'), (2.0, 'RACK2'), (3.0, 'RACK3')])
    assert is_region_pending_selection(reg, {'RACK1', 'RACK2'}) is True
    assert is_region_pending_selection(reg, {'RACK1', 'RACK2', 'RACK3'}) is True


def test_is_pending_false_when_exactly_one_candidate_checked():
    reg = _region(0, [(1.0, 'RACK1'), (2.0, 'RACK2')])
    assert is_region_pending_selection(reg, {'RACK2'}) is False


# ---------------------------------------------------------------------------
# app._gather_name_selections（グローバル決定＋ファイル別UIのマージ）
# ---------------------------------------------------------------------------

def test_gather_name_selections_uses_global_decision_for_single_match():
    _set_global_checked('RACK2')
    analysis = {'regions': [_region(5, [(1.0, 'RACK1'), (2.0, 'RACK2')])]}
    result = app._gather_name_selections('a.dxf', analysis)
    assert result == {('a.dxf', 5): ['RACK2']}


def test_gather_name_selections_single_candidate_requires_global_check():
    """候補1件の領域は「領域一覧」チェックが唯一のゲート。
    チェックが無ければ結果に含まれない（無所属）。"""
    analysis = {'regions': [_region(1, [(1.0, 'CTC')])]}

    _set_global_checked()
    assert app._gather_name_selections('a.dxf', analysis) == {}

    _set_global_checked('CTC')
    assert app._gather_name_selections('a.dxf', analysis) == {('a.dxf', 1): ['CTC']}


def test_gather_name_selections_no_candidate_checked_is_excluded():
    """候補が2件以上あっても、いずれもチェックされていなければ
    「領域の確認」のフォールバックUIにも回さず、無所属のまま除外する
    （2026-07-30、ユーザー指定）。"""
    _set_global_checked()
    analysis = {'regions': [_region(2, [(1.0, 'RACK1'), (2.0, 'RACK2')])]}
    result = app._gather_name_selections('a.dxf', analysis)
    assert result == {}


def test_gather_name_selections_falls_back_to_checkbox_ui_when_multiple_checked():
    """候補のうちチェック済みが2件以上ある領域は、「領域の確認」の
    チェックボックス状態（rc_<fname>_<reg_id>_<i>）から集める。インデックス
    `i` は `pending_candidates_for_region()` で絞り込んだ後の並び順に対応する。"""
    _set_global_checked('RACK1', 'RACK2')
    st.session_state['rc_a.dxf_2_1'] = True  # 絞り込み後index1（'RACK2'）をチェック
    analysis = {'regions': [_region(2, [(1.0, 'RACK1'), (2.0, 'RACK2')])]}
    result = app._gather_name_selections('a.dxf', analysis)
    assert result == {('a.dxf', 2): ['RACK2']}


def test_gather_name_selections_checkbox_ui_index_matches_filtered_list():
    """候補3件のうち2件だけチェックされている場合、絞り込み後のリストの
    先頭（index0）が実際にはどの候補になるかを確認する回帰テスト。"""
    _set_global_checked('RACK1', 'RACK3')  # RACK2は未チェック→選択肢から除外
    st.session_state['rc_a.dxf_9_0'] = True  # 絞り込み後index0（'RACK1'）をチェック
    analysis = {'regions': [
        _region(9, [(1.0, 'RACK1'), (2.0, 'RACK2'), (3.0, 'RACK3')]),
    ]}
    result = app._gather_name_selections('a.dxf', analysis)
    assert result == {('a.dxf', 9): ['RACK1']}


def test_gather_name_selections_no_candidates_region_excluded():
    """候補0件の領域はここでは何も返さない
    （`ref_designator.build_named_regions()` 等の自動 "no name" 命名に委ねる）。"""
    _set_global_checked()
    analysis = {'regions': [_region(3, [])]}
    assert app._gather_name_selections('a.dxf', analysis) == {}


def test_gather_name_selections_mixed_regions_in_one_file():
    """1ファイル内に「グローバル決定（1件一致）」「候補1件未チェック」
    「フォールバックUI（2件以上一致）」「未チェックで除外」「候補0件」が
    混在していても、それぞれ正しく処理される。"""
    _set_global_checked('SYSTEM I/F BOX', 'RACK1', 'RACK2')
    st.session_state['rc_a.dxf_10_0'] = True  # フォールバックUIで絞り込み後index0選択
    analysis = {'regions': [
        _region(0, [(1.0, 'SYSTEM I/F BOX'), (2.0, 'OTHER')]),  # グローバル決定(1件一致)
        _region(1, [(1.0, 'CTC')]),                              # 候補1件・未チェック→除外
        _region(10, [(1.0, 'RACK1'), (2.0, 'RACK2')]),           # フォールバックUI(2件一致)
        _region(15, [(1.0, 'UNCHECKED_A'), (2.0, 'UNCHECKED_B')]),  # 誰もチェックせず→除外
        _region(20, []),                                         # 候補0件
    ]}
    result = app._gather_name_selections('a.dxf', analysis)
    assert result == {
        ('a.dxf', 0): ['SYSTEM I/F BOX'],
        ('a.dxf', 10): ['RACK1'],
    }


# ---------------------------------------------------------------------------
# app._global_checked_region_names（grc_<name> セッション状態からの集合構築）
# ---------------------------------------------------------------------------

def test_global_checked_region_names_reads_grc_prefixed_keys():
    st.session_state['grc_RACK1'] = True
    st.session_state['grc_RACK2'] = False
    st.session_state['grc_CTC'] = True
    st.session_state['unrelated_key'] = True
    assert app._global_checked_region_names() == {'RACK1', 'CTC'}


def test_global_checked_region_names_empty_when_nothing_checked():
    assert app._global_checked_region_names() == set()
