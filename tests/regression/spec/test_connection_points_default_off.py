"""仕様: 接続点（円を含むブロック）による「橋渡しの停止」と「領域の除外」は既定で無効（v3.24.0）。

**受入条件**（2026-11-21、`EE6888-604-01A.dxf` の `ESC BOX` が領域にならない件。ユーザー決定: 既定値で無効化）:

- Given 既定の設定、When `analyze_dxf_regions()`、Then 端子（`ASM_*` ブロックの円）が境界の隙間・境界上にある
  機器の箱が領域になる: `EE6888-604-01A` の `ESC BOX`（15.5%）、`EE6892-612-01B` の `MPD RACK1`（5.9%）、
  `EE6892-611-01B` の `ＲＦ ＰＳ２ Ｒ１（ＲＦＳ－１３３０Ｎ）`（12.7%）。
- Given `exclude_connection_point_regions`・`stop_bridge_at_connection_points` を True（従来の動き）、
  Then 同じ図面で上の領域は出ない（設定で元に戻せる）。
- 2つのキーの既定値は False。

**以前どうだったか**: ①縦線の隙間に接続点があると橋渡ししない、②境界上に接続点があると領域ごと除外、の両方が
既定で有効で、ESC BOX は右辺の隙間4か所に端子が載るため閉じず、閉じても②で落ちた。円の大きさでは配線ループの
接続点と箱の端子を区別できなかった（除外された104領域の円は ESC BOX の端子と同じ半径 0.5）。
sample-dxf 252ファイルで両方を外して変わるのは5件で、消える・名称が変わる領域は0件（追加のみ）。
他の領域の分割・名称は変わらない（このテストは追加される領域だけを固定する）。

設定は DXF-extract-labels の `app.py` と同じ（`detect_minor_regions`・`local_rotation`）。
"""
import os
import sys
import unicodedata

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tests'))

from model.region_detector import DEFAULT_REGION_CONFIG, analyze_dxf_regions  # noqa: E402
from sample_data import require_sample  # noqa: E402

pytestmark = pytest.mark.slow

APP_CFG = {'detect_minor_regions': True, 'local_rotation': True}
LEGACY_CFG = {**APP_CFG, 'exclude_connection_point_regions': True, 'stop_bridge_at_connection_points': True}

# (ファイル, 名称〔NFKC〕, 面積比%, bbox (x0, x1, y0, y1) または None)
EXPECTED = [
    ('EE6888-604-01A.dxf', 'ESC BOX', 15.5, (296.6, 415.6, 93.6, 370.5)),
    ('EE6892-612-01B.dxf', 'MPD RACK1', 5.9, None),
    ('EE6892-611-01B.dxf', 'RF PS2 R1(RFS-1330N)', 12.7, None),
]


def _nfkc(text):
    return unicodedata.normalize('NFKC', text or '').replace(' ', '')


def _find(analysis, name, pct, bbox):
    for r in analysis['regions']:
        if _nfkc(r.get('default_name')) != _nfkc(name) or abs(r['area_pct'] - pct) > 0.1:
            continue
        if bbox is not None:
            xs = [p[0] for p in r['polygon']]
            ys = [p[1] for p in r['polygon']]
            got = (min(xs), max(xs), min(ys), max(ys))
            if any(abs(a - b) > 0.1 for a, b in zip(got, bbox)):
                continue
        return r
    return None


def _analyze(name, cfg):
    path = require_sample(name)
    if path is None:
        pytest.skip('sample-dxf が無い環境')
    analysis = analyze_dxf_regions(path, cfg)
    assert analysis['error'] is None
    return analysis


def test_connection_point_settings_default_to_off():
    assert DEFAULT_REGION_CONFIG['exclude_connection_point_regions'] is False
    assert DEFAULT_REGION_CONFIG['stop_bridge_at_connection_points'] is False


@pytest.mark.parametrize('name,region_name,pct,bbox', EXPECTED, ids=[e[0] for e in EXPECTED])
def test_boxes_with_connection_points_are_regions_by_default(name, region_name, pct, bbox):
    assert _find(_analyze(name, APP_CFG), region_name, pct, bbox) is not None


@pytest.mark.parametrize('name,region_name,pct,bbox', EXPECTED, ids=[e[0] for e in EXPECTED])
def test_legacy_behaviour_is_restored_by_the_two_settings(name, region_name, pct, bbox):
    assert _find(_analyze(name, LEGACY_CFG), region_name, pct, bbox) is None
