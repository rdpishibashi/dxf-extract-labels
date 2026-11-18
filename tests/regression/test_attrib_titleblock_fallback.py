"""タイトル欄の属性（ATTRIB）を図番・タイトル判定の手がかりにするフォールバック（2026-11-18）。

**不具合の経緯**: DWG から変換した古い図面（EE3298-637-02E ほか5件。`sample-dxf/dataset/ME25-7103-0_Cu_R1chamber/393_ケーブル/`）は、
タイトル欄ブロック `ULVAC_TITLE_05` が図番・タイトルを属性（`DWG.NO{..}`・`TITLE{..}`）で持つ。`extract_labels()` は
TEXT/MTEXT しか集めないので図番候補が0件になり、図番からタイトル欄を特定する処理（`main_group`・枠外文字の除外）に
入れないまま、図面枠ブロックの区画記号 `F` をタイトルにしていた（Cable-checker の比較タブに「F」グループが出て発覚）。

**仕様**: 表示中の TEXT/MTEXT でも、レイヤー off/frozen を無視した探し直しでも図番候補が1件も無いときだけ、
表示中の INSERT の表示中の属性を図番・タイトルの判定用ラベルに加える。出力ラベルには入れない。
図番が文字で取れる図面の結果は変わらない。
"""
import glob
import os
import sys
import tempfile

import ezdxf
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model.extract_labels import extract_labels, get_title_and_subtitle  # noqa: E402

DWG_NO, TITLE = 'EE9999-001-01A', 'SAMPLE TITLE'


def _add_frame(doc, name='FRAME'):
    """図面枠4辺（lineweight=100, color=7）・図面情報枠の境界線・枠外の区画記号 `F`（実データの ZZWAKUA3Y-ULVAC_03 を模す）。"""
    block = doc.blocks.new(name=name)
    for p1, p2 in [((0, 0), (200, 0)), ((200, 0), (200, 100)), ((200, 100), (0, 100)), ((0, 100), (0, 0))]:
        block.add_line(p1, p2, dxfattribs={'lineweight': 100, 'color': 7})
    block.add_line((130, 0), (130, 30))
    block.add_line((130, 30), (200, 30))
    block.add_text('F', dxfattribs={'insert': (205, 50), 'height': 3})
    return block


def _add_titleblock(doc, name='TITLEBLOCK'):
    """固定の項目名（文字）だけを持つタイトル欄ブロック。図番・タイトルは INSERT の属性で与える。"""
    block = doc.blocks.new(name=name)
    block.add_text('DWG.No', dxfattribs={'insert': (150, 20), 'height': 3})
    block.add_text('TITLE', dxfattribs={'insert': (100, 50), 'height': 3})
    block.add_attdef('DWG.NO', (170, 25), dxfattribs={'height': 3})
    block.add_attdef('TITLE', (140, 50), dxfattribs={'height': 3})
    return block


def _doc_with_attrib_titleblock(invisible_insert=False, invisible_attribs=False):
    doc = ezdxf.new()
    msp = doc.modelspace()
    _add_frame(doc)
    _add_titleblock(doc)
    msp.add_blockref('FRAME', (0, 0))
    ins = msp.add_blockref('TITLEBLOCK', (0, 0), dxfattribs={'invisible': 1} if invisible_insert else None)
    attrs = {'height': 3, **({'invisible': 1} if invisible_attribs else {})}
    ins.add_attrib('DWG.NO', DWG_NO, (170, 25), dxfattribs=attrs)
    ins.add_attrib('TITLE', TITLE, (140, 50), dxfattribs=attrs)
    msp.add_text('R10', dxfattribs={'insert': (10, 10), 'height': 3})
    return doc


def _save(doc):
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as f:
        path = f.name
    doc.saveas(path)
    return path


def _extract(doc, name=DWG_NO + '.dxf'):
    path = _save(doc)
    try:
        return extract_labels(path, extract_drawing_numbers_option=True, extract_title_option=True, original_filename=name)
    finally:
        os.remove(path)


def test_title_and_drawing_number_come_from_attributes_when_no_text_has_a_drawing_number():
    """守るもの: 図番が文字に無く属性にだけある図面で、図番・タイトルが属性から取れる（区画記号 `F` をタイトルにしない）。"""
    _, info = _extract(_doc_with_attrib_titleblock())
    assert info['main_drawing_number'] == DWG_NO
    assert info['title'] == TITLE


def test_attribute_values_are_not_added_to_output_labels():
    """守るもの: 属性は判定用だけ。出力ラベルには入れない（出力は表示中の TEXT/MTEXT のまま）。"""
    labels, _ = _extract(_doc_with_attrib_titleblock())
    assert DWG_NO not in labels and TITLE not in labels
    assert 'R10' in labels


def test_get_title_and_subtitle_wrapper_uses_the_fallback():
    path = _save(_doc_with_attrib_titleblock())
    try:
        assert get_title_and_subtitle(path, original_filename=DWG_NO + '.dxf')[0] == TITLE
    finally:
        os.remove(path)


@pytest.mark.parametrize('kwargs', [{'invisible_insert': True}, {'invisible_attribs': True}], ids=['insert', 'attribs'])
def test_invisible_insert_or_attributes_are_not_used(kwargs):
    """守るもの: 非表示の INSERT・非表示の属性は手がかりにしない（図番は取れず、タイトルは属性の値にならない）。"""
    _, info = _extract(_doc_with_attrib_titleblock(**kwargs))
    assert info['main_drawing_number'] is None
    assert info['title'] != TITLE


def test_fallback_is_not_used_when_a_text_has_a_drawing_number():
    """守るもの: 図番が文字で取れる図面では属性は使わない（結果は従来のまま。属性側の別の図番に引きずられない）。"""
    doc = ezdxf.new()
    msp = doc.modelspace()
    block = _add_frame(doc, 'VISIBLE_TB')
    block.add_text('TITLE', dxfattribs={'insert': (100, 50), 'height': 3})
    block.add_text('TEXT TITLE', dxfattribs={'insert': (140, 50), 'height': 3})
    block.add_text('EE1111-001-01A', dxfattribs={'insert': (170, 25), 'height': 3})
    msp.add_blockref('VISIBLE_TB', (0, 0))
    _add_titleblock(doc)
    ins = msp.add_blockref('TITLEBLOCK', (1000, 0))
    ins.add_attrib('DWG.NO', 'EE2222-002-02B', (1170, 25), dxfattribs={'height': 3})
    ins.add_attrib('TITLE', 'ATTRIB TITLE', (1140, 50), dxfattribs={'height': 3})
    _, info = _extract(doc, 'EE1111-001-01A.dxf')
    assert info['main_drawing_number'] == 'EE1111-001-01A'
    assert info['all_drawing_numbers'] == ['EE1111-001-01A']
    assert info['title'] == 'TEXT TITLE'


# ---- 実データ（DWG から変換した古い図面6件と、図番が文字の図面） ----

DATA_DIR = os.path.join(PROJECT_ROOT, 'sample-dxf', 'dataset', 'ME25-7103-0_Cu_R1chamber', '393_ケーブル')

# 図番が属性にしかない6件。タイトルは `ケーブル製作図`／`ケ－ブル製作図`（全角ハイフン）の表記ゆれを含む（実測 2026-11-18）
ATTRIB_ONLY = {
    'EE3298-568-01A': 'ケーブル製作図',
    'EE3298-574-01A': 'ケーブル製作図',
    'EE3298-574-02A': 'ケ－ブル製作図',
    'EE3298-574-30A': 'ケーブル製作図',
    'EE3298-574-33A': 'ケ－ブル製作図',
    'EE3298-637-02E': '線番設定表',
}
# 図番が文字で取れる図面（結果は従来のまま）
TEXT_DRAWING_NUMBER = {'EE3298-637-03R': '線番設定表', 'EE3298-534-07D': 'ケ－ブル製作図'}


@pytest.mark.slow
@pytest.mark.parametrize('name, title', {**ATTRIB_ONLY, **TEXT_DRAWING_NUMBER}.items())
def test_real_data_titles_and_drawing_numbers(name, title):
    path = os.path.join(DATA_DIR, name + '.dxf')
    assert os.path.isfile(path), f'{path} が見つかりません（skip しない）'
    labels, info = extract_labels(path, extract_drawing_numbers_option=True, extract_title_option=True,
                                  original_filename=name + '.dxf')
    assert info['main_drawing_number'] == name and info['title'] == title
    assert (name in labels) == (name in TEXT_DRAWING_NUMBER)     # 属性だけの図面は図番を出力ラベルに出さない
