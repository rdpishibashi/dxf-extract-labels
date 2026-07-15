"""合体親（union parent）のN子一般化（v1.9.1）の合成DXFによる end-to-end 確認。

`_detect_union_parents` が2子限定から任意N子（N>=2）に一般化されたことで、
3分割以上に仕切られた矩形（実データ: `EE6888-650-01C.dxf` の FL1F①②③、
`tests/regression/test_region_extraction.py` で確認）が正しく解消できることを、
分割の向き（左右どちらが再分割されるか・上下どちらが再分割されるか）・
子の数（4分割）・無関係な入れ子ノイズ候補の混入、という直積の各パターンで
合成DXFにより検証する。前提: 分割線分はいずれも途切れなし（実データのケースは
別途 `tests/regression/test_region_extraction.py` で確認）。

実データでの確認は `tests/regression/test_region_extraction.py` 側（FL1F・
DE5434-563-03A）で行う。本ファイルは合成DXFで直積の残りのセルを埋める。
"""
import os
import sys
import tempfile

import ezdxf
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from utils.region_detector import analyze_dxf_regions, DEFAULT_REGION_CONFIG  # noqa: E402

FRAME_LW = DEFAULT_REGION_CONFIG['frame_lineweight']
FRAME_COLOR = DEFAULT_REGION_CONFIG['frame_color']
BOUNDARY_LW = DEFAULT_REGION_CONFIG['region_lineweight']
BOUNDARY_COLOR = DEFAULT_REGION_CONFIG['region_color']


def _add_rect_lines(msp, x0, y0, x1, y1, lineweight, color):
    for p1, p2 in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                   ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
        msp.add_line(p1, p2, dxfattribs={'lineweight': lineweight, 'color': color})


def _add_frame(msp, x0, y0, x1, y1):
    _add_rect_lines(msp, x0, y0, x1, y1, FRAME_LW, FRAME_COLOR)


def _add_boundary_rect(msp, x0, y0, x1, y1):
    _add_rect_lines(msp, x0, y0, x1, y1, BOUNDARY_LW, BOUNDARY_COLOR)


def _add_divider(msp, p1, p2):
    msp.add_line(p1, p2, dxfattribs={'lineweight': BOUNDARY_LW, 'color': BOUNDARY_COLOR})


def _add_label(msp, text, x, y):
    msp.add_text(text, dxfattribs={'insert': (x, y)})


def _analyze(build_fn):
    doc = ezdxf.new()
    msp = doc.modelspace()
    build_fn(msp)
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as f:
        path = f.name
    doc.saveas(path)
    try:
        return analyze_dxf_regions(path, DEFAULT_REGION_CONFIG)
    finally:
        os.remove(path)


def _names(a):
    return sorted(r['default_name'] for r in a['regions'])


def _region_by_name(a, name):
    return next(r for r in a['regions'] if r['default_name'] == name)


# 外枠 (0,0)-(200,100)、frame_area=20000 を共通で使う。
FX0, FY0, FX1, FY1 = 0, 0, 200, 100


def _build_split_left(msp):
    """入れ子3子: 縦仕切りで左右分割 → 左列をさらに横仕切りで上下分割。
    (FL1Fは右列側が再分割されるパターン。本テストはその鏡像=左列側)"""
    _add_frame(msp, FX0, FY0, FX1, FY1)
    _add_boundary_rect(msp, FX0, FY0, FX1, FY1)  # 外周自体も境界線(lw25/color2)を持つ
    _add_divider(msp, (100, 0), (100, 100))   # 縦仕切り(全高)
    _add_divider(msp, (0, 50), (100, 50))     # 横仕切り(左半分のみ)
    _add_label(msp, 'LEFT LOWER', 20, 3)
    _add_label(msp, 'LEFT UPPER', 20, 53)
    _add_label(msp, 'RIGHT SIDE', 140, 3)


def _build_split_bottom(msp):
    """入れ子3子: 横仕切りで上下分割 → 下段をさらに縦仕切りで左右分割。"""
    _add_frame(msp, FX0, FY0, FX1, FY1)
    _add_boundary_rect(msp, FX0, FY0, FX1, FY1)  # 外周自体も境界線(lw25/color2)を持つ
    _add_divider(msp, (0, 50), (200, 50))     # 横仕切り(全幅)
    _add_divider(msp, (100, 0), (100, 50))    # 縦仕切り(下半分のみ)
    _add_label(msp, 'BOTTOM LEFT', 20, 3)
    _add_label(msp, 'BOTTOM RIGHT', 120, 3)
    _add_label(msp, 'TOP SIDE', 20, 53)


def _build_split_top(msp):
    """入れ子3子: 横仕切りで上下分割 → 上段をさらに縦仕切りで左右分割。"""
    _add_frame(msp, FX0, FY0, FX1, FY1)
    _add_boundary_rect(msp, FX0, FY0, FX1, FY1)  # 外周自体も境界線(lw25/color2)を持つ
    _add_divider(msp, (0, 50), (200, 50))     # 横仕切り(全幅)
    _add_divider(msp, (100, 50), (100, 100))  # 縦仕切り(上半分のみ)
    _add_label(msp, 'BOTTOM SIDE', 20, 3)
    _add_label(msp, 'TOP LEFT', 20, 53)
    _add_label(msp, 'TOP RIGHT', 120, 53)


def _build_split_4way(msp):
    """4子: 縦仕切り1本(全高)で左右分割し、さらに左右それぞれを異なる高さの
    横仕切りで上下分割（4分割）。

    左右の横仕切りを**同じ高さ**（例 y=50 で共通）にすると、`_merge_collinear`
    が縦仕切りを挟んで両側の横仕切りセグメントを「同一レベル・隣接」とみなし
    1本の連続した横線に結合し直してしまい、縦仕切りとの接続点(T字)が消えて
    中ほど交差と同じ状態（4分割にならず外周1面に潰れる）になることを確認した。
    左右で異なる高さ（y=50/y=30）にすることで、この結合を避け、各横仕切りが
    縦仕切りに独立したT字で接続する。面探索は「線分の端点が相手の線分に乗る
    箇所（角・T字）のみ」で接続し中ほど交差では繋がない設計（無関係な部品の
    中央交差による誤結合を防ぐため）のため、1本の縦線と1本の横線が中ほどで
    交差する真のX字交点は、そもそもこのアルゴリズムの対応範囲外（検証済み）。"""
    _add_frame(msp, FX0, FY0, FX1, FY1)
    _add_boundary_rect(msp, FX0, FY0, FX1, FY1)  # 外周自体も境界線(lw25/color2)を持つ
    _add_divider(msp, (100, 0), (100, 100))   # 縦仕切り(全高、左右2分割)
    _add_divider(msp, (0, 50), (100, 50))     # 横仕切り(左側のみ、y=50)
    _add_divider(msp, (100, 30), (200, 30))   # 横仕切り(右側のみ、y=30。左と異なる高さ)
    _add_label(msp, 'QUADRANT A', 20, 3)      # 左下 (0,0)-(100,50) 25%
    _add_label(msp, 'QUADRANT B', 20, 53)     # 左上 (0,50)-(100,100) 25%
    _add_label(msp, 'QUADRANT C', 120, 3)     # 右下 (100,0)-(200,30) 15%
    _add_label(msp, 'QUADRANT D', 120, 33)    # 右上 (100,30)-(200,100) 35%


def _build_split_with_island(msp):
    """2子（縦仕切りで左右分割）+ 左側の内部に無関係な独立した小矩形(island)が
    同居するケース。`DE5434-563-03A.dxf`（実データ）で島候補が誤って正しい
    兄弟候補を子から除外させていた不具合の合成版（`_detect_union_parents` の
    面積降順貪欲選択で解消済み）。"""
    _add_frame(msp, FX0, FY0, FX1, FY1)
    _add_boundary_rect(msp, FX0, FY0, FX1, FY1)  # 外周自体も境界線(lw25/color2)を持つ
    _add_divider(msp, (100, 0), (100, 100))   # 縦仕切り(全高)。左右2子。
    _add_boundary_rect(msp, 20, 30, 60, 70)   # 左側内部の無関係な独立小矩形(island)
    _add_label(msp, 'LEFT SIDE', 20, 3)
    _add_label(msp, 'RIGHT SIDE', 120, 3)
    _add_label(msp, 'ISLAND BOX', 25, 33)


@pytest.mark.parametrize('build_fn,expected_names,area_pct_by_name,removed_union_bbox', [
    (_build_split_left, {'LEFT LOWER', 'LEFT UPPER', 'RIGHT SIDE'},
     {'LEFT LOWER': 25.0, 'LEFT UPPER': 25.0, 'RIGHT SIDE': 50.0}, (0, 0, 200, 100)),
    (_build_split_bottom, {'BOTTOM LEFT', 'BOTTOM RIGHT', 'TOP SIDE'},
     {'BOTTOM LEFT': 25.0, 'BOTTOM RIGHT': 25.0, 'TOP SIDE': 50.0}, (0, 0, 200, 100)),
    (_build_split_top, {'BOTTOM SIDE', 'TOP LEFT', 'TOP RIGHT'},
     {'BOTTOM SIDE': 50.0, 'TOP LEFT': 25.0, 'TOP RIGHT': 25.0}, (0, 0, 200, 100)),
    (_build_split_4way, {'QUADRANT A', 'QUADRANT B', 'QUADRANT C', 'QUADRANT D'},
     {'QUADRANT A': 25.0, 'QUADRANT B': 25.0, 'QUADRANT C': 15.0, 'QUADRANT D': 35.0},
     (0, 0, 200, 100)),
], ids=['split-left-column', 'split-bottom-row', 'split-top-row', '4-way-grid'])
def test_n_child_union_parent_resolved_and_named(
        build_fn, expected_names, area_pct_by_name, removed_union_bbox):
    """各分割パターンで、合体親（外周）が除去され、全ての子が個別領域として
    正しい面積比・名称で検出されること。"""
    a = _analyze(build_fn)
    assert a['error'] is None
    names = set(r['default_name'] for r in a['regions'])
    assert expected_names <= names, f'期待した名称が不足: {expected_names - names}'
    for name, expected_pct in area_pct_by_name.items():
        r = _region_by_name(a, name)
        pct = 100.0 * r['area'] / a['frame_area']
        assert abs(pct - expected_pct) < 1.0, f'{name} の面積比が想定外: {pct:.1f}%'
    # 合体親（外周そのもの）が独立領域として残っていないこと
    x0, y0, x1, y1 = removed_union_bbox
    for r in a['regions']:
        xs = [p[0] for p in r['polygon']]
        ys = [p[1] for p in r['polygon']]
        is_full_outer = (min(xs) <= x0 + 1 and max(xs) >= x1 - 1
                         and min(ys) <= y0 + 1 and max(ys) >= y1 - 1)
        assert not is_full_outer, f'合体親（外周）が除去されず残っている: {r["default_name"]!r}'


def test_island_noise_does_not_block_sibling_pair_detection():
    """左側内部に無関係な独立小矩形(island)が同居していても、左右2子が正しく
    合体親から解消されること（`DE5434-563-03A.dxf` 実データ相当の合成版）。"""
    a = _analyze(_build_split_with_island)
    assert a['error'] is None
    names = _names(a)
    assert 'LEFT SIDE' in names
    assert 'RIGHT SIDE' in names
    assert 'ISLAND BOX' in names
    left = _region_by_name(a, 'LEFT SIDE')
    right = _region_by_name(a, 'RIGHT SIDE')
    assert abs(100.0 * left['area'] / a['frame_area'] - 50.0) < 1.0
    assert abs(100.0 * right['area'] / a['frame_area'] - 50.0) < 1.0
    for r in a['regions']:
        xs = [p[0] for p in r['polygon']]
        ys = [p[1] for p in r['polygon']]
        is_full_outer = (min(xs) <= 1 and max(xs) >= 199 and min(ys) <= 1 and max(ys) >= 99)
        assert not is_full_outer, f'合体親（外周）が除去されず残っている: {r["default_name"]!r}'
