"""viewer のラベル色替え: 抽出されるラベルが、すべて個別に色を変えられる文字として図面に存在する（v3.17.0）。

守るもの（受入条件）:
  Given 実データの図面（sample-dxf の curated フォルダ。ブロック内の文字・入れ子・色指定付き MTEXT・寸法を含むもの）
  When  `render_base_svg()` で図面を描く
  Then  領域検出の `analysis['labels']` と抽出パイプラインの `data['labels']` の全ラベルが
        `BaseRender.label_classes` に対応する（対応しないラベルは重ね書きにフォールバックし、位置のずれの原因になる）

背景: ブロック内の文字だけをモデル空間へ移し、文字ごとの目印の色で描く方式（`model/region_preview.py`）。
文字が抽出と描画で食い違うと、ハイライトできない・別の文字が光る。curated 49図面すべてで未対応0件を確認済み
（2026-10-25）。ここでは重い図面を含む代表6件を回す。
文字以外の線のパスは ezdxf の描画ごとに僅かに揺れるので、実データのパス座標の完全一致は要求しない
（合成データでの一致は tests/unit/test_region_preview.py が守る）。
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from model import region_preview as rp  # noqa: E402
from model.ref_designator import normalize_label  # noqa: E402
from model.region_detector import DEFAULT_REGION_CONFIG  # noqa: E402
from tests import dxf_cache  # noqa: E402

SAMPLE_DIR = os.path.join(PROJECT_ROOT, 'sample-dxf')


from tests.sample_data import find_sample as _sample  # noqa: E402

SAMPLES = [
    'EE2685-335-01D.dxf',     # ブロック内に文字（JZB_0001）と寸法
    'EE5322-455-01B.dxf',     # 寸法ブロック（*D…）が多数。全ブロックから文字を消すと寸法値が消える
    'EE6888-650-01C.dxf',
    'EE6892-039-82A.dxf',
    pytest.param('DE3527-553-10E.dxf', marks=pytest.mark.slow),
    pytest.param('EE6868-500-01C.dxf', marks=pytest.mark.slow),   # 約1万ラベル
]


@pytest.mark.parametrize('name', SAMPLES)
def test_every_extracted_label_has_a_css_class(name):
    path = _sample(name)
    assert path, f'{name} が sample-dxf 配下（再帰）にない（データ欠落。skip にしない）'
    with open(path, 'rb') as f:
        base = rp.render_base_svg(f.read())
    analysis = dxf_cache.analyze_dxf_regions(path, dict(DEFAULT_REGION_CONFIG))
    extracted = dxf_cache.extract_ref_designator_data(
        path, frame_lineweight=DEFAULT_REGION_CONFIG['frame_lineweight'])

    def missing(labels):
        return [lab for lab in labels
                if (normalize_label(lab[0]), round(lab[1], 1), round(lab[2], 1)) not in base.label_classes]

    assert analysis.get('labels'), '領域検出のラベルが空（テストが何も確かめていない）'
    assert missing(analysis['labels']) == []
    assert missing(extracted['labels']) == []
