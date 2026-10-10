"""入れ子の領域で、同じ距離の名称候補は内側の領域だけに残す（2026-10-10）。

**以前の不具合**: `EE3294-039-05A.dxf` で、53%の外側領域の切り欠き辺と5.6%の内側領域の
下辺が同じ線にあり、内側領域のラベル `SENSOR I/F B.D-Ⅲ` が両方の領域から同じ距離
（1.2）になった。距離が同じ場合は「優劣なし」としてどちらからも外さないため、外側の領域
の候補にも残っていた。
**保証すること**: 距離が同じでも、相手が自分に完全に含まれる内側の領域なら外側の候補から
外す。含まれない（別々の）重なる領域は従来どおり外さない。
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from model.region_detector import _remove_overlap_claimed_candidates  # noqa: E402


def _region(poly, cands):
    return {'polygon': poly, 'name_candidates': list(cands),
            'default_name': cands[0][1], 'default_name_tier': 1, '_tier_by_text': {}}


OUTER = [(0, 0), (100, 0), (100, 100), (0, 100)]
INNER = [(50, 20), (90, 20), (90, 60), (50, 60)]


def test_equal_distance_label_removed_from_outer_region():
    outer = _region(OUTER, [(2.0, 'OUTER BOX'), (1.2, 'INNER BOX')])
    inner = _region(INNER, [(1.2, 'INNER BOX')])
    _remove_overlap_claimed_candidates([outer, inner])
    assert [t for _, t in outer['name_candidates']] == ['OUTER BOX']
    assert [t for _, t in inner['name_candidates']] == ['INNER BOX']


def test_equal_distance_between_non_nested_overlapping_regions_is_kept():
    """片方がもう片方に完全に含まれない重なり（相互にはみ出す）では従来どおり外さない。"""
    a = _region([(0, 0), (60, 0), (60, 60), (0, 60)], [(1.2, 'SHARED')])
    b = _region([(40, 40), (100, 40), (100, 100), (40, 100)], [(1.2, 'SHARED')])
    _remove_overlap_claimed_candidates([a, b])
    assert [t for _, t in a['name_candidates']] == ['SHARED']
    assert [t for _, t in b['name_candidates']] == ['SHARED']
