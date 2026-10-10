"""回転図面では、枠内に別の大きい領域があっても横ギャップ橋渡しを行う（2026-10-10）。

**以前の不具合**: 回転図面向けの横ギャップ橋渡しパスは「枠に面積閾値を満たす候補が
1件も無い場合」だけ発動していた。`EE6333-610-07A.dxf` は枠に65.4%の領域があるため
このパスが素通りし、内包する6頂点の領域（約30%、`#10D0`/`#DA4`/`#1184`/`#1186`/`#1185`
を辺とする。上辺 y=352.9 が部品記号で3か所途切れている）が検出されなかった。
**保証すること**: 回転図面（ラベルの過半数が90°）では全枠で橋渡しを行い、この領域が
検出される。回転図面でなければ従来どおり橋渡ししない。
"""
import glob
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from model.region_detector import analyze_dxf_regions  # noqa: E402

_FILES = glob.glob(os.path.join(PROJECT_ROOT, 'sample-dxf', '*', 'EE6333-610-07A.dxf'))
EXPECTED = [(44.9, 132.2), (795.2, 132.2), (795.2, 352.9), (118.9, 352.9), (118.9, 259.1), (44.9, 259.1)]


@pytest.mark.slow
@pytest.mark.skipif(not _FILES, reason='サンプル DXF が無い')
def test_inner_six_vertex_region_is_detected_in_rotated_drawing():
    a = analyze_dxf_regions(_FILES[0])
    assert a['error'] is None
    # 36.0%（コーナーは6つ。#10D0・#1184 を辺に含む）。境界線 y=169.3 の仕切りで下に
    # 6.3%・上に29.7%の領域にも分かれて検出される（polygon には共線上の頂点が入る）。
    inner = [r for r in a['regions'] if 30 < r['area_pct'] < 40 and len(r['corners']) == 6]
    assert len(inner) == 1
    for ex, ey in EXPECTED:
        assert any(abs(ex - x) < 0.5 and abs(ey - y) < 0.5 for x, y in inner[0]['corners'])
