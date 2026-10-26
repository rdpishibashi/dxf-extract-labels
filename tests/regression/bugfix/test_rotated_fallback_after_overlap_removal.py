"""回転図面で、重なり判定で候補が空になった領域にも横エッジ近傍の探索を行う（2026-10-10）。

**以前の不具合**: `EE6892-612-01B.dxf` の領域0（62.9%）は縦エッジの候補 `SB-1A(R1R)` を一度は持つが、
内側の領域4が同じ名前をより近い位置で持つため重なり判定で外れ、候補が空（無名）になった。
横エッジ近傍の探索は「縦エッジに候補が1件も無いとき」だけだったため、この領域の内側の
`ＲＡＣＫ`（下辺から2.5）は候補にならなかった。
**保証すること**: 重なり判定の後に候補が空になった（面積条件を満たす）領域にも横エッジ近傍の探索を行い、
`ＲＡＣＫ` が候補になる。`EE6333-610-07A.dxf` の65.4%の領域は、下辺から12.0の `RACK` が
横エッジ探索の最大距離（`horizontal_name_max_dist`、既定15）で候補になる。
"""
import glob
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from model.region_detector import analyze_dxf_regions  # noqa: E402


from tests.sample_data import pool_exists, require_sample  # noqa: E402


def _sample(name):
    # sample-dxf があるのに無ければ AssertionError（skip にしない）
    return require_sample(name)


@pytest.mark.slow
@pytest.mark.skipif(_sample('EE6892-612-01B.dxf') is None, reason='サンプル DXF が無い')
def test_rack_named_for_region_emptied_by_overlap_removal():
    a = analyze_dxf_regions(_sample('EE6892-612-01B.dxf'))
    big = next(r for r in a['regions'] if 60 < r['area_pct'] < 65)
    assert [t for _d, t in big['name_candidates']][:1] == ['ＲＡＣＫ']
    inner = next(r for r in a['regions'] if 27 < r['area_pct'] < 29)
    assert 'ＲＡＣＫ' not in [t for _d, t in inner['name_candidates']]  # 近い側（領域0）が持つ


@pytest.mark.slow
@pytest.mark.skipif(_sample('EE6333-610-07A.dxf') is None, reason='サンプル DXF が無い')
def test_rack_within_15_of_horizontal_edge_is_a_candidate():
    """2026-10-26: 面積比の分母を用紙サイズ別の固定値に変えた（A1、旧値×1.026）ため、
    対象領域の面積比は 65.4% → 67.1%（窓 65–66 → 67–68）。"""
    a = analyze_dxf_regions(_sample('EE6333-610-07A.dxf'))
    big = next(r for r in a['regions'] if 67 < r['area_pct'] < 68)
    assert [t for _d, t in big['name_candidates']] == ['RACK']


@pytest.mark.slow
@pytest.mark.skipif(_sample('EE6333-610-07A.dxf') is None, reason='サンプル DXF が無い')
def test_dc19a4_is_a_region_name_candidate_in_07a():
    """「英字+数字+英字+数字」（`DC19A4`）は機器符号候補ではなく領域名候補（2026-10-10、
    `label_classifier.NON_DESIGNATOR_PATTERNS`。戻すときはそのタプルを空にする）。

    2026-10-26: 面積比の分母を用紙サイズ別の固定値に変えた（A1、旧値×1.026）ため、
    対象領域の面積比は 29.7% → 30.5%（窓 29–30 → 30–31）。"""
    a = analyze_dxf_regions(_sample('EE6333-610-07A.dxf'))
    r = next(r for r in a['regions'] if 30 < r['area_pct'] < 31)
    assert 'DC19A4' in [t for _d, t in r['name_candidates']]
