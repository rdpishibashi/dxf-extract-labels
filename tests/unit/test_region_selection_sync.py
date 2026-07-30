"""app.py の領域名称選択チェックボックス同期コールバック
（`_on_change_radio` / `_on_change_none`）の単体テスト。

`st.session_state` を辞書に差し替えることで Streamlit ランタイム無しに
テストする（`app.py` は `st.session_state[...]` の形でのみアクセスするため、
辞書互換で成立する）。
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st  # noqa: E402
import app  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_session_state():
    st.session_state = {}
    yield
    st.session_state = {}


def _region(fname, reg_id, polygon, candidates):
    """テスト用の領域dictを作る。"""
    return {'id': reg_id, 'polygon': polygon, 'name_candidates': list(candidates)}


def _set_analyses(regions_by_file):
    st.session_state['region_analyses'] = {
        fname: {'regions': regions} for fname, regions in regions_by_file.items()
    }


def _set_checkbox(fname, reg_id, idx, value):
    st.session_state[f"rc_{fname}_{reg_id}_{idx}"] = value


def _get_checkbox(fname, reg_id, idx):
    return st.session_state.get(f"rc_{fname}_{reg_id}_{idx}")


# ---------------------------------------------------------------------------
# _on_change_none — OFF伝播（2026-07-26 追加）
# ---------------------------------------------------------------------------

def test_on_change_none_turns_off_matching_candidate_in_other_region():
    """領域Aで「選択しない」をONにすると、領域Aの候補テキストと一致する
    他領域（重ならない）のチェックボックスもOFFになる。"""
    poly_a = [(0, 0), (10, 0), (10, 10), (0, 10)]
    poly_b = [(100, 0), (110, 0), (110, 10), (100, 10)]  # 重ならない
    region_a = _region('f.dxf', 0, poly_a, [(1.0, 'X'), (2.0, 'Y')])
    region_b = _region('f.dxf', 1, poly_b, [(1.5, 'X'), (3.0, 'Z')])
    _set_analyses({'f.dxf': [region_a, region_b]})

    # 事前状態: 領域Bで X が採用済み（以前 _on_change_radio 経由で同期された想定）
    _set_checkbox('f.dxf', 1, 0, True)   # B: X = ON
    _set_checkbox('f.dxf', 1, 1, False)  # B: Z = OFF

    _set_checkbox('f.dxf', 0, 'none', True)
    app._on_change_none('f.dxf', 0, 2, {'X', 'Y', 'Z'})

    assert _get_checkbox('f.dxf', 1, 0) is False  # B の X が OFF になる
    assert _get_checkbox('f.dxf', 1, 1) is False  # 元々 OFF の Z は変化なし


def test_on_change_none_does_not_propagate_to_overlapping_region_same_file():
    """同一ファイル内で重なりのある領域同士は、選択しないのOFF伝播も行わない
    （`_on_change_radio` の重なり除外と対称的な扱い）。"""
    poly_outer = [(0, 0), (100, 0), (100, 100), (0, 100)]
    poly_inner = [(10, 10), (20, 10), (20, 20), (10, 20)]  # 完全内包
    region_outer = _region('f.dxf', 0, poly_outer, [(1.0, 'X'), (2.0, 'Y')])
    region_inner = _region('f.dxf', 1, poly_inner, [(1.5, 'X'), (3.0, 'Z')])
    _set_analyses({'f.dxf': [region_outer, region_inner]})

    _set_checkbox('f.dxf', 1, 0, True)  # inner: X = ON

    _set_checkbox('f.dxf', 0, 'none', True)
    app._on_change_none('f.dxf', 0, 2, {'X', 'Y', 'Z'})

    assert _get_checkbox('f.dxf', 1, 0) is True  # 重なり領域なので変化しない


def test_on_change_none_turns_on_none_when_other_region_left_with_nothing_selected():
    """伝播の結果、他領域の候補が全てOFFになった場合、その領域自身の
    「領域選択しない」もONになる（2026-07-26 ユーザー指摘の追加要件）。"""
    poly_a = [(0, 0), (10, 0), (10, 10), (0, 10)]
    poly_b = [(100, 0), (110, 0), (110, 10), (100, 10)]
    region_a = _region('f.dxf', 0, poly_a, [(1.0, 'X'), (2.0, 'Y')])
    # B の唯一の候補が X（Aの候補と一致）
    region_b = _region('f.dxf', 1, poly_b, [(1.5, 'X'), (3.0, 'W')])
    _set_analyses({'f.dxf': [region_a, region_b]})

    _set_checkbox('f.dxf', 1, 0, True)   # B: X = ON（唯一の選択）
    _set_checkbox('f.dxf', 1, 1, False)  # B: W = OFF

    _set_checkbox('f.dxf', 0, 'none', True)
    app._on_change_none('f.dxf', 0, 2, {'X', 'Y', 'W'})

    assert _get_checkbox('f.dxf', 1, 0) is False
    assert _get_checkbox('f.dxf', 1, 1) is False
    assert st.session_state.get('rc_f.dxf_1_none') is True  # Bの「選択しない」がON


def test_on_change_none_does_not_turn_on_none_when_other_candidate_remains_selected():
    """他領域に、今回の伝播と無関係な候補が別途ONのまま残っていれば、
    その領域の「領域選択しない」はONにしない（全滅時のみONにする）。"""
    poly_a = [(0, 0), (10, 0), (10, 10), (0, 10)]
    poly_b = [(100, 0), (110, 0), (110, 10), (100, 10)]
    region_a = _region('f.dxf', 0, poly_a, [(1.0, 'X'), (2.0, 'Y')])
    region_b = _region('f.dxf', 1, poly_b, [(1.5, 'X'), (3.0, 'W')])
    _set_analyses({'f.dxf': [region_a, region_b]})

    _set_checkbox('f.dxf', 1, 0, False)  # B: X = OFF
    _set_checkbox('f.dxf', 1, 1, True)   # B: W = ON（Aとは無関係な独自選択）

    _set_checkbox('f.dxf', 0, 'none', True)
    app._on_change_none('f.dxf', 0, 2, {'X', 'Y', 'W'})

    assert _get_checkbox('f.dxf', 1, 0) is False
    assert _get_checkbox('f.dxf', 1, 1) is True  # 無関係な選択は維持
    assert st.session_state.get('rc_f.dxf_1_none') is not True  # 何か選ばれているのでONにしない


def test_on_change_none_propagates_across_files():
    """OFF伝播は _on_change_radio のON伝播と同じく全ファイル横断で働く。"""
    poly_a = [(0, 0), (10, 0), (10, 10), (0, 10)]
    poly_c = [(0, 0), (10, 0), (10, 10), (0, 10)]  # 別ファイルなので座標が同じでも無関係
    region_a = _region('f1.dxf', 0, poly_a, [(1.0, 'X')])
    region_c = _region('f2.dxf', 5, poly_c, [(2.0, 'X'), (3.0, 'W')])
    _set_analyses({'f1.dxf': [region_a], 'f2.dxf': [region_c]})

    _set_checkbox('f2.dxf', 5, 0, True)  # f2側: X = ON

    _set_checkbox('f1.dxf', 0, 'none', True)
    app._on_change_none('f1.dxf', 0, 1, {'X', 'W'})

    assert _get_checkbox('f2.dxf', 5, 0) is False  # 別ファイルでも X は OFF になる


def test_on_change_none_includes_single_candidate_regions():
    """候補が1件のみ（選択肢なし・自動確定）の領域も OFF 伝播の対象に含める
    （`_on_change_radio` のON伝播とは異なる扱い。2026-07-26 ユーザー報告:
    `DE5027-563-03A.dxf` で候補1件のみの領域が複数連続するケース
    〈領域21-24=`NXSAF FX1`〉で、1つを「選択しない」にしても兄弟領域に
    伝播しなかった）。伝播先が候補1件のみでも、その候補が一致していれば
    OFFにし、結果として候補が全滅すれば「選択しない」もONにする。"""
    poly_a = [(0, 0), (10, 0), (10, 10), (0, 10)]
    poly_b = [(100, 0), (110, 0), (110, 10), (100, 10)]
    poly_c = [(200, 0), (210, 0), (210, 10), (200, 10)]
    region_a = _region('f.dxf', 0, poly_a, [(1.0, 'X')])  # 発火元も候補1件のみ
    region_b = _region('f.dxf', 1, poly_b, [(1.5, 'X')])  # 候補1件のみ
    region_c = _region('f.dxf', 2, poly_c, [(2.0, 'X')])  # 候補1件のみ
    _set_analyses({'f.dxf': [region_a, region_b, region_c]})

    _set_checkbox('f.dxf', 0, 0, True)  # A: X = ON（自動確定）
    _set_checkbox('f.dxf', 1, 0, True)  # B: X = ON（自動確定）
    _set_checkbox('f.dxf', 2, 0, True)  # C: X = ON（自動確定）

    _set_checkbox('f.dxf', 0, 'none', True)
    app._on_change_none('f.dxf', 0, 1, {'X'})

    assert _get_checkbox('f.dxf', 0, 0) is False
    assert _get_checkbox('f.dxf', 1, 0) is False  # 候補1件のみでも OFF になる
    assert _get_checkbox('f.dxf', 2, 0) is False
    assert st.session_state.get('rc_f.dxf_1_none') is True  # 全滅なので選択しないもON
    assert st.session_state.get('rc_f.dxf_2_none') is True


def test_on_change_none_noop_when_not_checked():
    """コールバックの ck 自体が False のとき（チェックが外れた側の呼び出し
    等）は何もしない。"""
    poly_a = [(0, 0), (10, 0), (10, 10), (0, 10)]
    region_a = _region('f.dxf', 0, poly_a, [(1.0, 'X')])
    _set_analyses({'f.dxf': [region_a]})
    _set_checkbox('f.dxf', 0, 'none', False)
    app._on_change_none('f.dxf', 0, 1, {'X'})
    # 例外なく完走すればOK（副作用が起きていないことは他のテストで別途確認済み）


# ---------------------------------------------------------------------------
# _on_change_radio — 既存のON伝播（無回帰確認）
# ---------------------------------------------------------------------------

def test_on_change_radio_still_propagates_on_and_clears_none():
    poly_a = [(0, 0), (10, 0), (10, 10), (0, 10)]
    poly_b = [(100, 0), (110, 0), (110, 10), (100, 10)]
    region_a = _region('f.dxf', 0, poly_a, [(1.0, 'X'), (2.0, 'Y')])
    region_b = _region('f.dxf', 1, poly_b, [(1.5, 'X'), (3.0, 'Z')])
    _set_analyses({'f.dxf': [region_a, region_b]})

    _set_checkbox('f.dxf', 0, 'none', True)  # 事前に「選択しない」がONだったとする
    _set_checkbox('f.dxf', 0, 0, True)  # X をクリック
    app._on_change_radio('f.dxf', 0, 0, 2, 'X', {'X', 'Y', 'Z'})

    assert _get_checkbox('f.dxf', 0, 1) is False       # 同一領域の他候補はOFF
    assert st.session_state.get('rc_f.dxf_0_none') is False  # 選択しないが解除される
    assert _get_checkbox('f.dxf', 1, 0) is True        # 他領域のXがON
    assert _get_checkbox('f.dxf', 1, 1) is False        # 他領域のZはOFF
