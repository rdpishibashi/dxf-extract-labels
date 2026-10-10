"""領域の外にあるラベルを名称候補にする予備ルール（Tier3）は使わない（2026-10-10、ユーザー決定）。

**受入条件**: 領域の内側に上辺・下辺の名称候補が1つも無いとき、領域の外にあるラベルが
近くにあっても名称候補にならない（候補は空）。結果を見て戻す可能性があるため、
戻す場合はこのテストを置き換えること。
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from model.region_detector import region_name_candidates  # noqa: E402

POLY = [(0, 0), (100, 0), (100, 50), (0, 50)]


def test_label_outside_region_is_not_a_candidate():
    cands, _ = region_name_candidates(POLY, [('OUTSIDE LABEL', 10.0, 53.0)])
    assert cands == []


def test_label_inside_near_top_edge_is_a_candidate():
    cands, tiers = region_name_candidates(POLY, [('INSIDE LABEL', 10.0, 47.0)])
    assert [t for _, t in cands] == ['INSIDE LABEL']
    assert tiers['INSIDE LABEL'] == 2
