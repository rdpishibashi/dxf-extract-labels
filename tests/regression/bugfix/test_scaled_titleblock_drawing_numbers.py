"""不具合再発防止: 拡大配置したタイトルブロックで流用元図番・タイトルが取れない。

**識別子**: 2026-11-06、DXF-diff-manager で `EE7319-300-01A.dxf`（流用先）と
`EE6094-300-01C.dxf`（流用元）の差分が `EE7319-300-01A_vs_none.dxf` になった件。

**以前どう壊れていたか（再現条件）**: 図面全体を 10 倍の座標（A2 = 5940×4200）で描き、
タイトルブロックのブロックを `xscale=10` で INSERT した図面。流用元図番・タイトルの
判定に使う距離しきい値（`SOURCE_LABEL_PROXIMITY=80`・`TITLE_PROXIMITY_X=80` 等）は
等倍のタイトルブロックを前提にした図面単位の値のため、「流用元図番」ラベルから
流用元図番までの距離（実データで約173）が届かず、流用元図番が None（完全新規扱い）、
タイトルも None になっていた。

**修正後に保証したいこと**: タイトルブロックの拡大率で判定用の座標を正規化し、
等倍と 10 倍で図番・流用元図番・タイトル・サブタイトルが同じになる。
出力ラベルの座標は図面座標のまま（正規化しない）。

合成DXF（tmp_path）のみを使う。座標は実データのブロック `JZB_0001` の局所座標に合わせた。
"""
import os
import sys

import ezdxf
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from model.extract_labels import extract_labels  # noqa: E402

# (テキスト, ブロック内の局所座標)
TITLEBLOCK_TEXTS = [
    ('TITLE', (512.83, 43.87)),
    ('REVISION', (547.0, 48.5)),
    ('流用元図番', (409.8, 60.37)),
    ('EE6094-300-01C', (427.0, 58.2)),
    ('DWG No.', (487.73, 15.80)),
    ('EE7319-300-01A', (533.0, 15.5)),
    ('MPD RACK1', (547.0, 36.6)),
    ('WARNING_LABEL', (547.0, 27.3)),
]


def _write_dxf(path, scale):
    doc = ezdxf.new('R2010')
    block = doc.blocks.new('TB')
    for text, (x, y) in TITLEBLOCK_TEXTS:
        block.add_mtext(text, dxfattribs={'insert': (x, y), 'char_height': 2.5})
    doc.modelspace().add_blockref('TB', (0, 0), dxfattribs={'xscale': scale, 'yscale': scale})
    doc.saveas(path)


@pytest.mark.parametrize('scale', [1.0, 10.0])
def test_titleblock_scale_does_not_change_drawing_numbers_or_title(tmp_path, scale):
    path = str(tmp_path / 'EE7319-300-01A.dxf')
    _write_dxf(path, scale)

    _, info = extract_labels(path, extract_drawing_numbers_option=True, extract_title_option=True)

    assert info['main_drawing_number'] == 'EE7319-300-01A'
    assert info['source_drawing_number'] == 'EE6094-300-01C'
    assert info['title'] == 'MPD RACK1'
    assert info['subtitle'] == 'WARNING_LABEL'


def test_output_label_coordinates_stay_in_drawing_units(tmp_path):
    path = str(tmp_path / 'EE7319-300-01A.dxf')
    _write_dxf(path, 10.0)

    labels, _ = extract_labels(path, extract_drawing_numbers_option=True, extract_title_option=True,
                               include_coordinates=True)

    coords = {text: (x, y) for text, x, y in labels}
    assert coords['TITLE'] == pytest.approx((5128.3, 438.7))
