"""対辺補完 `_complete_rectangles()` の単体テスト（合成データ。2026-10-17）。

守る対象: 四角形の1辺が途切れていても、対辺と他の2辺がそろい、途切れた辺の**両方の角**から線が出ていれば
その辺を補う。片角しか無い・円（接続点）がある・対辺の範囲がずれる場合は補わない（無関係な線を繋がない）。
実例 `EE3273-039-06B.dxf` の `SENSOR I/F B.D-Ⅲ` 枠（左辺が部品箱で途切れ、上側の片は長さ2未満）。
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model.region_detector import _complete_rectangles, _raw_axis_segments  # noqa: E402

# 横線 y=0 と y=30（x 0..20）、右辺 x=20（y 0..30）。左辺 x=0 は y 0..10 と y 20..30 の2片だけ（間 10〜20 が欠け）
H = [(0.0, 0.0, 20.0), (30.0, 0.0, 20.0)]
V_BASE = [(20.0, 0.0, 30.0), (0.0, 0.0, 10.0), (0.0, 20.0, 30.0)]


def test_missing_side_is_completed_when_both_corners_have_lines():
    Hm, Vm = _complete_rectangles(list(H), list(V_BASE), [], [])
    assert (0.0, 0.0, 30.0) in Vm or (0.0, 0.0, 30) in Vm or any(
        v[0] == 0.0 and v[1] == 0.0 and v[2] == 30.0 for v in Vm)
    assert Hm == H   # 横方向は増えない


def test_short_piece_counts_as_corner_evidence_via_raw_segments():
    """Vm には載らない長さ<snap の短片（raw_v だけにある）も角の証拠になる。"""
    v = [(20.0, 0.0, 30.0), (0.0, 0.0, 10.0)]          # 上側の片は短くて Vm に無い
    raw_v = [(0.0, 28.5, 30.0)]
    _, Vm = _complete_rectangles(list(H), v, [], raw_v)
    assert any(x == 0.0 and lo == 0.0 and hi == 30.0 for x, lo, hi in Vm)


def test_single_corner_is_not_enough():
    v = [(20.0, 0.0, 30.0), (0.0, 0.0, 10.0)]          # 上側の角に線が無い
    _, Vm = _complete_rectangles(list(H), v, [], [])
    assert Vm == v


def test_circle_on_the_gap_blocks_completion():
    _, Vm = _complete_rectangles(list(H), list(V_BASE), [], [], circles=[(0.0, 15.0)], band=0.05)
    assert Vm == V_BASE


def test_mismatched_opposite_sides_are_not_paired():
    h = [(0.0, 0.0, 20.0), (30.0, 0.0, 25.0)]           # 上辺の範囲が違う
    _, Vm = _complete_rectangles(h, list(V_BASE), [], [])
    assert Vm == V_BASE


def test_horizontal_side_is_completed_symmetrically():
    # 縦線 x=0 と x=30（y 0..20）、上辺 y=20 完全。下辺 y=0 は x 0..10 と 20..30 の2片（間が欠け）
    v = [(0.0, 0.0, 20.0), (30.0, 0.0, 20.0)]
    h = [(20.0, 0.0, 30.0), (0.0, 0.0, 10.0), (0.0, 20.0, 30.0)]
    Hm, Vm = _complete_rectangles(h, v, [], [])
    assert any(y == 0.0 and lo == 0.0 and hi == 30.0 for y, lo, hi in Hm)
    assert Vm == v


def test_raw_axis_segments_keeps_short_pieces_inside_frame_only():
    lines = [((0.0, 28.5), (0.0, 30.0), 'a'),      # 長さ1.5（snap 2.0 未満）→ 残す
             ((500.0, 0.0), (500.0, 10.0), 'b'),   # 枠外 → 捨てる
             ((1.0, 5.0), (5.0, 5.0), 'c'),        # 横
             ((0.0, 0.0), (0.0, 0.1), 'd')]        # 長さ0.1 < 0.3 → 捨てる
    raw_h, raw_v = _raw_axis_segments(lines, (-5.0, 100.0, -5.0, 100.0), 5.0)
    assert raw_v == [(0.0, 28.5, 30.0)]
    assert raw_h == [(5.0, 1.0, 5.0)]
