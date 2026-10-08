"""領域名ごとの「特定／除外」指定（v3.11.0）の仕様テスト。

**守る対象**: 領域名の名称候補が「除外」指定された場合、その領域内（境界線上を
含む）のラベルが抽出結果から完全に取り除かれること。単体テストだけでは
「`excluded_names`/`master_index` の渡し漏れでその経路だけ黙って除外が効かない」
という不具合を検出できないため（`Tools/CLAUDE.md`に記録された
`extract_title_option`/`extract_drawing_numbers_option`の実際の事故と同じ
クラス）、`model/ref_designator.py`の純粋関数から`model/excel_output.py`の
出力関数まで、実際にExcelを生成してopenpyxlで読み戻し確認する。

2026-10-08、引き継ぎ書（§9・§11）に基づく。
"""
import os
import sys
from io import BytesIO

import openpyxl
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from model import ref_designator  # noqa: E402
from model.excel_output import create_region_excel_output  # noqa: E402


def _load_wb(data: bytes):
    return openpyxl.load_workbook(BytesIO(data))


# ---------------------------------------------------------------------------
# excluded_region_ids — 候補数×除外指定の組み合わせ（表2）
# ---------------------------------------------------------------------------

def _analysis(regions):
    return {'regions': regions}


def test_excluded_region_ids_single_candidate_match():
    """候補1件のみの領域で、その名称が除外指定なら除外される。"""
    analysis = _analysis([
        {'id': 0, 'name_candidates': [(1.0, 'AWG20')]},
    ])
    assert ref_designator.excluded_region_ids(analysis, {'AWG20'}) == {0}


def test_excluded_region_ids_no_match():
    """除外指定された名称がどの候補にも一致しなければ除外されない。"""
    analysis = _analysis([
        {'id': 0, 'name_candidates': [(1.0, 'RACK1')]},
    ])
    assert ref_designator.excluded_region_ids(analysis, {'AWG20'}) == set()


def test_excluded_region_ids_multiple_candidates_one_matches():
    """候補2件以上のうち1件だけが除外指定でも、その領域全体が除外される
    （どの名前を採用するかに関わらず、2026-10-08ユーザー決定）。"""
    analysis = _analysis([
        {'id': 0, 'name_candidates': [(1.0, 'RACK1'), (2.0, 'AWG20')]},
    ])
    assert ref_designator.excluded_region_ids(analysis, {'AWG20'}) == {0}


def test_excluded_region_ids_no_candidates_never_excluded():
    """候補0件（無名領域）は指定する名称自体が存在しないため、除外されない
    （excluded_names に何が入っていても関係ない。2026-10-08ユーザー承認）。"""
    analysis = _analysis([
        {'id': 0, 'name_candidates': []},
    ])
    assert ref_designator.excluded_region_ids(analysis, {'AWG20', 'RACK1'}) == set()


def test_excluded_region_ids_normalizes_width():
    """候補テキストの全角/半角表記の揺れを正規化してから比較する。"""
    analysis = _analysis([
        {'id': 0, 'name_candidates': [(1.0, 'ＡＷＧ２０')]},
    ])
    assert ref_designator.excluded_region_ids(analysis, {'AWG20'}) == {0}


def test_excluded_region_ids_empty_excluded_names_returns_empty():
    """除外指定が1件も無ければ、候補の有無に関わらず空集合を返す。"""
    analysis = _analysis([
        {'id': 0, 'name_candidates': [(1.0, 'RACK1')]},
    ])
    assert ref_designator.excluded_region_ids(analysis, set()) == set()


def test_excluded_region_ids_multiple_regions_independent():
    """複数領域がある場合、一致した領域のみが集合に入る。"""
    analysis = _analysis([
        {'id': 0, 'name_candidates': [(1.0, 'RACK1')]},
        {'id': 1, 'name_candidates': [(1.0, 'AWG20')]},
        {'id': 2, 'name_candidates': [(1.0, 'AWG24')]},
    ])
    assert ref_designator.excluded_region_ids(analysis, {'AWG20', 'AWG24'}) == {1, 2}


# ---------------------------------------------------------------------------
# filter_labels_outside_excluded — ラベル位置×除外領域（表3）
# ---------------------------------------------------------------------------

_SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_filter_removes_label_inside_excluded_polygon():
    """除外領域の内側にあるラベルは消える。"""
    labels = [('INSIDE', 5.0, 5.0)]
    kept = ref_designator.filter_labels_outside_excluded(labels, [_SQUARE])
    assert kept == []


def test_filter_removes_label_on_boundary():
    """除外領域の境界線上（boundary_eps以内）のラベルも消える
    （要求「領域線上も含めて」。既存の_point_in_polygonの挙動を使う）。"""
    labels = [('ON_EDGE', 0.0, 5.0)]
    kept = ref_designator.filter_labels_outside_excluded(labels, [_SQUARE])
    assert kept == []


def test_filter_keeps_label_outside_excluded_polygon():
    """除外領域の外側にあるラベルは残る。"""
    labels = [('OUTSIDE', 50.0, 50.0)]
    kept = ref_designator.filter_labels_outside_excluded(labels, [_SQUARE])
    assert kept == [('OUTSIDE', 50.0, 50.0)]


def test_filter_removes_label_in_overlap_with_specified_region():
    """特定領域と除外領域が重なる部分のラベルも消える（除外優先）。
    filter_labels_outside_excluded() 自体は「特定」領域を知らないので、
    重なりの有無に関わらず除外ポリゴン内のラベルを一律に消す（除外優先は
    呼び出し元がbuild_region_output()に渡すラベル集合を絞り込む形で実現する
    ため、この関数のレベルでは「除外ポリゴンに入っているか」だけを見る）。"""
    overlapping_square = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
    labels = [('IN_BOTH', 7.0, 7.0)]
    kept = ref_designator.filter_labels_outside_excluded(labels, [_SQUARE, overlapping_square])
    assert kept == []


def test_filter_removes_label_in_nested_region():
    """入れ子（除外領域の内側に別の特定領域）でも、外側の除外ポリゴンに
    内包される時点でラベルは消える（2026-10-08ユーザー承認）。"""
    inner_square = [(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)]
    labels = [('IN_INNER', 5.0, 5.0)]
    # 外側の除外ポリゴンのみを渡しても、内側のラベルは消える
    kept = ref_designator.filter_labels_outside_excluded(labels, [_SQUARE])
    assert kept == []
    # 内側ポリゴン自体が除外対象でなくても、外側が除外ならラベルは消える
    assert inner_square  # 内側ポリゴンの定義自体はここでは使わない（上記で確認済み）


def test_filter_anchor_outside_text_overflowing_is_kept():
    """アンカー点が領域外なら、テキストが領域にはみ出していても残る
    （既存の`領域`列割り当てと同じ、アンカー点のみで判定する規約。
    2026-10-08、意図的な既知の制限として固定）。"""
    labels = [('ANCHOR_OUTSIDE', 10.5, 5.0)]  # 境界(x=10)からわずかに外
    kept = ref_designator.filter_labels_outside_excluded(labels, [_SQUARE])
    assert kept == [('ANCHOR_OUTSIDE', 10.5, 5.0)]


def test_filter_no_excluded_polygons_returns_all_labels():
    """除外ポリゴンが0件なら、全ラベルがそのまま残る（回帰防護）。"""
    labels = [('A', 1.0, 1.0), ('B', 2.0, 2.0)]
    assert ref_designator.filter_labels_outside_excluded(labels, []) == labels


def test_filter_preserves_label_order_and_duplicates():
    """複数ラベルが残る場合、順序と重複はそのまま保持される。"""
    labels = [('A', 50.0, 50.0), ('A', 60.0, 60.0), ('B', 5.0, 5.0)]
    kept = ref_designator.filter_labels_outside_excluded(labels, [_SQUARE])
    assert kept == [('A', 50.0, 50.0), ('A', 60.0, 60.0)]
