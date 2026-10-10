"""回転図面で縦エッジに候補が無い領域は、横エッジ近傍のラベルも領域名候補にする（2026-10-10）。

**受入条件**: 回転図面（`rotated_edge_roles` 指定）で、縦エッジ（下端/上端相当）の探索で
候補が1件も無く、かつ領域が面積条件を満たす（`horizontal_fallback=True`）ときだけ、
横エッジ（DXFの上下の辺）の内側近傍のラベルを候補にする（Tier3）。縦エッジに候補が
ある領域は変わらない。回転図面でなければ使わない。実例: `EE6333-610-07A.dxf` の
`MPD RACK3`、`EE6892-612-01B.dxf` の `DC PS3 R1 (…)`。
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from model.region_detector import region_name_candidates  # noqa: E402

POLY = [(0, 0), (100, 0), (100, 50), (0, 50)]
# ラベルは横エッジ（下辺 y=0）の内側 3、縦エッジ（x=0/100）からは遠い
BOTTOM_LABEL = [('MPD RACK3', 50.0, 3.0)]
ROLES = ('right', 'left')


def test_fallback_used_when_no_vertical_candidate():
    cands, tiers = region_name_candidates(
        POLY, BOTTOM_LABEL, rotated_edge_roles=ROLES, horizontal_fallback=True)
    assert [t for _d, t in cands] == ['MPD RACK3']
    assert tiers['MPD RACK3'] == 3


def test_fallback_off_by_default():
    cands, _ = region_name_candidates(POLY, BOTTOM_LABEL, rotated_edge_roles=ROLES)
    assert cands == []


def test_not_used_for_non_rotated_drawings_with_flag_only():
    """通常図面では横エッジが元々 Tier1/2。回転指定が無ければフォールバックは働かない
    （この入力は下辺の内側3なので通常の Tier1 で候補になる）。"""
    cands, tiers = region_name_candidates(POLY, BOTTOM_LABEL, horizontal_fallback=True)
    assert tiers['MPD RACK3'] == 1


def test_vertical_candidates_take_precedence_and_fallback_is_skipped():
    labels = [('SIDE NAME', 97.0, 25.0), ('MPD RACK3', 50.0, 3.0)]  # 右の縦エッジの内側3
    cands, tiers = region_name_candidates(
        POLY, labels, rotated_edge_roles=ROLES, horizontal_fallback=True)
    assert [t for _d, t in cands] == ['SIDE NAME']
