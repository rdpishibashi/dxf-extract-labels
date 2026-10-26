"""領域の面積比の分母は「用紙サイズ別の固定値（図面枠 − 図面情報枠）」であること（2026-10-26 ユーザー指定）。

守る対象: `analyze_dxf_regions()` の `frame_area`（`area_pct`・`area_ratio` 判定の分母）は、最初の図面枠の
寸法（±2）が `PAPER_FRAME_TABLE` に一致すれば `W*H - 情報枠W*情報枠H`、一致しなければ図面枠の面積。
用紙サイズを図面から読み取らず、図面情報枠の検出も使わない。1ページ分の値で、ページを合算しない。
既定の最小面積（`area_ratio`）は 4%。

受入条件:
  - 表の5寸法（A3 385×277・A2 559×400・A1 796×554・旧テンプレート 770×554 と 1118×800）で、分母が
    95,445 / 212,400 / 429,784 / 401,380 / 869,200 になる。許容誤差 ±2 の内側は表、外側は図面枠の面積
  - 表に無い寸法の図面枠（例 350×554）は従来どおり図面枠の面積
  - 合成 DXF の `area_pct` は `area / 新分母`（旧分母より大きい）
  - 複数ページ（図面枠2つ）でも分母は1つで、ページ数に比例して増えない
  - 既定の `area_ratio` は 0.04、`config.py` の `AREA_RATIO_PERCENT` は 4
  - 実データ `EE6888-639-01A.dxf`（A3）: 既定設定で `TMP CONTR ** (EI-R04M)`（4.19%）が検出され、
    `SYSTEM I/F BOX **` は 5.15%
"""
import os
import sys
import tempfile

import ezdxf
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from model.region_detector import (  # noqa: E402
    DEFAULT_REGION_CONFIG, PAPER_FRAME_TABLE, analyze_dxf_regions, effective_frame_area)
from config import RegionDetectionConfig  # noqa: E402
from tests.sample_data import pool_exists, require_sample  # noqa: E402

FRAME_LW = DEFAULT_REGION_CONFIG['frame_lineweight']
FRAME_COLOR = DEFAULT_REGION_CONFIG['frame_color']
BOUNDARY_LW = DEFAULT_REGION_CONFIG['region_lineweight']
BOUNDARY_COLOR = DEFAULT_REGION_CONFIG['region_color']


@pytest.mark.parametrize('w,h,expected', [
    (385, 277, 95445),        # A3: 385*277 - 175*64
    (559, 400, 212400),       # A2
    (796, 554, 429784),       # A1
    (770, 554, 401380),       # 旧テンプレート（情報枠 350×72）
    (1118, 800, 869200),      # 旧テンプレート
])
def test_table_dimensions_use_fixed_denominator(w, h, expected):
    assert effective_frame_area((10, 10 + w, 20, 20 + h)) == expected


def test_table_is_computed_from_frame_minus_info_frame():
    """表の分母は数値の手書きではなく `W*H - 情報枠W*情報枠H`（表の各行と一致）。"""
    for pw, ph, iw, ih, _note in PAPER_FRAME_TABLE:
        assert effective_frame_area((0, pw, 0, ph)) == pw * ph - iw * ih


@pytest.mark.parametrize('dw,dh,uses_table', [
    (2.0, 0.0, True), (0.0, -2.0, True), (-2.0, 2.0, True),     # 許容誤差 ±2 の境界は表
    (2.1, 0.0, False), (0.0, 2.1, False),                       # 外側は図面枠の面積
])
def test_tolerance_boundary(dw, dh, uses_table):
    w, h = 385 + dw, 277 + dh
    area = effective_frame_area((0, w, 0, h))
    assert area == (95445 if uses_table else w * h)


@pytest.mark.parametrize('w,h', [(350, 554), (175, 277), (200, 100), (420, 297)])
def test_unknown_dimensions_fall_back_to_frame_area(w, h):
    assert effective_frame_area((5, 5 + w, 7, 7 + h)) == w * h


def _rect(msp, x0, y0, x1, y1, lw, color):
    for p1, p2 in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                   ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
        msp.add_line(p1, p2, dxfattribs={'lineweight': lw, 'color': color})


def _analyze(frames, regions, config=None):
    """frames: [(x0, y0, x1, y1), ...]、regions: [(x0, y0, x1, y1, 名称), ...]"""
    doc = ezdxf.new()
    msp = doc.modelspace()
    for f in frames:
        _rect(msp, *f, FRAME_LW, FRAME_COLOR)
    for (x0, y0, x1, y1, name) in regions:
        _rect(msp, x0, y0, x1, y1, BOUNDARY_LW, BOUNDARY_COLOR)
        msp.add_text(name, dxfattribs={'insert': (x0 + 5, y0 + 3), 'height': 2.5})
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as f:
        path = f.name
    doc.saveas(path)
    try:
        return analyze_dxf_regions(path, config or dict(DEFAULT_REGION_CONFIG))
    finally:
        os.remove(path)


def test_synthetic_a3_area_pct_uses_fixed_denominator():
    """385×277 の図面枠に 200×120 の領域: area_pct = 24000/95445（旧分母 106645 なら 22.5%）。"""
    a = _analyze([(25, 10, 410, 287)], [(40, 30, 240, 150, 'TEST REGION')])
    assert a['error'] is None
    assert a['frame_area'] == 95445
    r = next(r for r in a['regions'] if r['default_name'] == 'TEST REGION')
    assert r['area_pct'] == pytest.approx(100.0 * 24000 / 95445, abs=0.01)
    assert r['area_pct'] > 100.0 * 24000 / 106645 + 1.0


def test_synthetic_unknown_frame_keeps_frame_area():
    """表に無い寸法（420×297）の図面枠は従来どおり図面枠の面積が分母。"""
    a = _analyze([(0, 0, 420, 297)], [(15, 20, 215, 140, 'TEST REGION')])
    assert a['error'] is None
    assert a['frame_area'] == 420 * 297
    r = next(r for r in a['regions'] if r['default_name'] == 'TEST REGION')
    assert r['area_pct'] == pytest.approx(100.0 * 24000 / (420 * 297), abs=0.01)


def test_multi_page_denominator_is_one_page():
    """図面枠が2つ（2ページ）でも分母は1ページ分（合算しない）。"""
    a = _analyze([(25, 10, 410, 287), (525, 10, 910, 287)],
                 [(40, 30, 240, 150, 'PAGE ONE'), (540, 30, 740, 150, 'PAGE TWO')])
    assert a['error'] is None
    assert a['frame_area'] == 95445
    for name in ('PAGE ONE', 'PAGE TWO'):
        r = next(r for r in a['regions'] if r['default_name'] == name)
        assert r['area_pct'] == pytest.approx(100.0 * 24000 / 95445, abs=0.01)


def test_default_min_area_is_4_percent():
    """既定の最小面積は 4%（整数%比較: 四捨五入して4以上）。4.19% の領域は採用され、3.35% の領域は採用されない。"""
    assert DEFAULT_REGION_CONFIG['area_ratio'] == 0.04
    assert RegionDetectionConfig.AREA_RATIO_PERCENT == 4
    # 分母 95445: 4000 → 4.19%、3200 → 3.35%（四捨五入で3）
    a = _analyze([(25, 10, 410, 287)],
                 [(40, 30, 90, 110, 'FOUR PERCENT'), (150, 30, 200, 94, 'THREE THIRD')])
    names = {r['default_name'] for r in a['regions']}
    assert 'FOUR PERCENT' in names          # 50×80 = 4000 → 4.19%
    assert 'THREE THIRD' not in names       # 50×64 = 3200 → 3.35%（四捨五入で4%未満）


@pytest.mark.slow
@pytest.mark.skipif(not pool_exists(), reason='共有 sample-dxf が無い環境')
def test_real_data_a3_tmp_contr_detected_by_default():
    """`EE6888-639-01A.dxf`（A3）: 5%では SYSTEM I/F BOX だけだった図面で、TMP CONTR（4.19%）も検出される。"""
    path = require_sample('EE6888-639-01A.dxf')
    a = analyze_dxf_regions(path)
    assert a['error'] is None
    assert a['frame_area'] == 95445
    by_name = {r['default_name']: r['area_pct'] for r in a['regions']}
    assert by_name['SYSTEM I/F BOX **'] == pytest.approx(5.15, abs=0.02)
    assert by_name['TMP CONTR ** (EI-R04M)'] == pytest.approx(4.19, abs=0.02)
