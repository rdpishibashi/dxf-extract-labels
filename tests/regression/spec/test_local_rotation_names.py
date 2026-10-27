"""図面の一部だけが90°回転して描かれた領域の名称探索（2026-10-27、ユーザー指示。`EE6892-610-01B.dxf`）。

守る対象: 図面全体は回転していない（回転ラベルが過半に満たない）が、ある領域の中身は左に90°回転して
描かれていて、領域名がその領域の右辺（回転した図面の「下辺」）のそばに書かれている場合、その名称を候補にする。

受入条件（`region_name_candidates(label_angles=…)`。`label_angles` は `cfg['local_rotation']` のときだけ作られる）:
  - 領域内のラベルの過半が±90°回転していて、回転側の辺（+90°なら右が下端相当）の最近傍ラベルが
    通常の辺（下辺・上辺）の最近傍ラベルより近いときだけ、回転図面と同じ辺の役割にする
  - 通常の辺のほうが近いとき（図面全体を含む大きい領域の `RACK` など）は従来どおり
  - `label_angles` が無ければ従来どおり（他プロジェクト・`local_rotation` なしは変わらない）
  - 回転した領域が複数あっても、上下を合わせた外形（名称が内側の領域に取られる）は名称を持たない
  - デバイス（4辺が揃った閉じた長方形）が仕切り線を隠している場合、線を補って領域を長方形にする
    （`_device_gap_bridges()`。`local_rotation` のときだけ）。デバイスの辺は領域の境界にしない。
    補うのは、補った線の両端が縦の境界線に届き、デバイスの手前の途切れが `_DEVICE_LEAD_MAX` 以内のときだけ
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: E402
from model.region_detector import _device_gap_bridges, analyze_dxf_regions, region_name_candidates  # noqa: E402
from view.region_selection import build_region_cfg  # noqa: E402

SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]
FILLERS = [(f'x{i}', 20.0 + 10 * i, 50.0) for i in range(6)]   # 領域内の回転ラベル（名称候補にはならない短い語）


def _angles(labels, rotated=True):
    return {(t, round(x, 1), round(y, 1)): (90.0 if rotated else 0.0) for (t, x, y) in labels}


def test_rotated_region_name_beside_right_edge_is_a_candidate():
    labels = FILLERS + [('UNIT NAME', 98.0, 50.0)]
    cands, _ = region_name_candidates(SQUARE, labels, label_angles=_angles(labels))
    assert [t for _d, t in cands] == ['UNIT NAME']


def test_without_label_angles_the_right_edge_label_is_not_a_candidate():
    labels = FILLERS + [('UNIT NAME', 98.0, 50.0)]
    cands, _ = region_name_candidates(SQUARE, labels)
    assert cands == []


def test_normal_bottom_edge_wins_when_it_is_nearer():
    labels = FILLERS + [('UNIT NAME', 96.0, 50.0), ('BOTTOM NAME', 50.0, 3.0)]
    angles = _angles(labels)
    angles[('BOTTOM NAME', 50.0, 3.0)] = 0.0
    cands, _ = region_name_candidates(SQUARE, labels, label_angles=angles)
    assert [t for _d, t in cands] == ['BOTTOM NAME']


def test_not_rotated_majority_is_unchanged():
    labels = FILLERS + [('UNIT NAME', 98.0, 50.0)]
    cands, _ = region_name_candidates(SQUARE, labels, label_angles=_angles(labels, rotated=False))
    assert cands == []


def test_too_few_labels_do_not_switch_to_rotated_edges():
    labels = FILLERS[:2] + [('UNIT NAME', 98.0, 50.0)]
    cands, _ = region_name_candidates(SQUARE, labels, label_angles=_angles(labels))
    assert cands == []


def _find_sample(name):
    for root, _dirs, files in os.walk(os.path.join(PROJECT_ROOT, 'sample-dxf')):
        if name in files:
            return os.path.join(root, name)
    return None


@pytest.mark.slow
def test_real_data_610_rotated_part_gets_both_names():
    """610-01B: 回転した部分は、上の領域（13.4%）が `DC11E4`、下の帯（3.8%）が `SB-1A(R1R)`。
    上下を合わせた外形（17.7%）は名称を持たず出力に残らない。図面全体を含む領域の `RACK` は変わらない。
    `CATHODE`（`_COMMON_NOUNS` から外した）が 6.1% の領域の名称候補になる。"""
    path = _find_sample('EE6892-610-01B.dxf')
    if path is None:
        pytest.skip('サンプル DXF が無い')
    cfg = {**build_region_cfg(config.region_detection_config), 'detect_minor_regions': True}
    names = lambda r: {round(g['area_pct'], 1): [t for _d, t in g['name_candidates']] for g in r['regions']}  # noqa: E731
    before = names(analyze_dxf_regions(path, cfg))
    rotated = analyze_dxf_regions(path, {**cfg, 'local_rotation': True})
    after = names(rotated)
    assert 'DC11E4' not in sum(before.values(), [])
    assert after[13.9] == ['DC11E4'] and after[3.8] == ['SB-1A(R1R)']
    assert 17.7 not in after
    assert after[48.7] == ['RACK']
    assert after[6.1] == ['CATHODE']
    # デバイス（`A91`・`A8F`・`A90`・`BB4`、上辺の `1-CNDCB` など）が仕切り線・上辺を隠していても、2領域は長方形
    sides = {round(g['area_pct'], 1): len(g['polygon']) for g in rotated['regions']}
    assert sides[13.9] == 4 and sides[3.8] == 4


# --- デバイスが仕切り線を隠している場合の補完（`_device_gap_bridges`）---
# 縦の境界 x=0・x=60（y 0〜10）、仕切り線 y=5、デバイス x 12〜40 × y 0〜10（4辺）
_DEVICE_H = [(0, 12, 40), (10, 12, 40)]
_DEVICE_V = [(12, 0, 10), (40, 0, 10)]


def _device_case(left_end=10.0, right_start=40.0, with_boundaries=True):
    Hm = [(5.0, 0.0, left_end), (5.0, right_start, 60.0), *_DEVICE_H]
    Vf = list(_DEVICE_V) + ([(0.0, 0.0, 10.0), (60.0, 0.0, 10.0)] if with_boundaries else [])
    Hf = [(5.0, 0.0, left_end), (5.0, right_start, 60.0), *_DEVICE_H]
    return Hf, Vf, Hm


def test_device_hiding_a_divider_is_bridged_including_the_short_lead_gap():
    Hf, Vf, Hm = _device_case()      # 左の線は 10 で止まり、デバイスの左辺 12 の手前で途切れる（2）
    assert _device_gap_bridges(Hf, Vf, Hm) == [(5.0, 10.0, 40.0)]


def test_device_bridge_requires_both_ends_to_reach_a_vertical_boundary():
    Hf, Vf, Hm = _device_case(with_boundaries=False)
    assert _device_gap_bridges(Hf, Vf, Hm) == []


def test_device_bridge_does_not_cross_a_long_gap():
    Hf, Vf, Hm = _device_case(left_end=2.0)     # 左の線がデバイスの手前 10 で止まる（上限 6 を超える）
    assert _device_gap_bridges(Hf, Vf, Hm) == []


def test_device_bridge_requires_all_four_sides_of_the_device():
    Hf, Vf, Hm = _device_case()
    assert _device_gap_bridges(Hf[:-1], Vf, Hm[:-1]) == []   # 下辺（y=10）が無い
