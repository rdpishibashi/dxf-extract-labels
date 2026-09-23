"""不具合再発防止（bugfix regression）: 唯一のタイトルブロック（図面枠を
含むINSERT）がoff+frozenレイヤーに置かれている図面で、図番・タイトル・
図面枠が一切検出できなくなっていた回帰の修正を固定する。

**不具合の経緯**: 2026-09-16に `common_utils.is_invisible()` へレイヤー単位
（off/frozen）の非表示判定を追加した（`test_layer_off_frozen_label_excluded_
from_collected_labels` 等が検証）。この判定はエンティティのレイヤーが
off/frozenであれば常に除外するため、**唯一のタイトルブロック（図面枠を
内包するINSERT）自体がoff/frozenレイヤーに置かれている図面**では、
図面枠・図番・タイトルの手がかりが一切残らず、抽出できなくなっていた
（実データ `EE5322-455-02A.dxf`/`EE5322-455-18A.dxf` で発覚、2026-09-23
ユーザー報告）。

**修正方針**: `is_invisible(e, check_layer=False)` で、エンティティ自身の
`invisible`属性のみを見るフォールバック探索を追加した。表示中のエンティティ
だけで図番・図面枠の手がかりが1件も見つからない場合に限りこのフォール
バックを使う。**出力するラベル自体は常に表示中のエンティティのみに限定
する**——フォールバックは図番・タイトル・図面枠を見つけるためだけに使う。

対象モジュール: `model/extract_labels.py`（`extract_labels()`）、
`model/ref_designator.py`（`collect_in_frame_labels()`）、
`model/region_detector.py`（`analyze_dxf_regions()`）。
"""
import os
import sys
import tempfile

import ezdxf
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model.extract_labels import extract_labels  # noqa: E402
from model import ref_designator  # noqa: E402
from model.region_detector import analyze_dxf_regions  # noqa: E402


def _add_titleblock_block(doc, name):
    """図面枠4辺（lineweight=100, color=7）と、タイトル項目名・図番・
    タイトル文字列を持つブロック定義を追加する（実データのタイトルブロック
    INSERTを模したもの）。

    図面情報枠（タイトルブロック）自身の境界線（右下の小さい矩形、左辺
    x=130・上辺y=30）を図面枠4辺とは別に持たせる——実データのタイトル
    ブロックは図面枠全体とは別に自分専用の矩形境界を持つため。これが無いと
    `_detect_info_frame_bbox()` が境界線を特定できず、フォールバック
    （ラベルbbox+マージン）にも失敗して図面枠全体を情報枠とみなしてしまう
    （2026-09-23、本テスト作成時に実際に踏んだ落とし穴）。"""
    block = doc.blocks.new(name=name)
    for p1, p2 in [((0, 0), (200, 0)), ((200, 0), (200, 100)),
                   ((200, 100), (0, 100)), ((0, 100), (0, 0))]:
        block.add_line(p1, p2, dxfattribs={'lineweight': 100, 'color': 7})
    # 図面情報枠自身の境界線（右下の矩形。右辺・底辺は図面枠と共有、
    # 左辺・上辺のみ追加すれば `_detect_info_frame_bbox()` が特定できる）
    block.add_line((130, 0), (130, 30))
    block.add_line((130, 30), (200, 30))
    # 図面情報枠（タイトルブロック）検出用のタイトル項目名（右下寄り）
    block.add_text('DWG.No', dxfattribs={'insert': (150, 20)})
    block.add_text('DATE', dxfattribs={'insert': (150, 15)})
    block.add_text('NAME', dxfattribs={'insert': (150, 10)})
    block.add_text('TITLE', dxfattribs={'insert': (100, 50)})
    block.add_text('SAMPLE TITLE', dxfattribs={'insert': (140, 50)})
    block.add_text('EE9999-001-01A', dxfattribs={'insert': (170, 25)})
    return block


def _save_and_load(doc):
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as f:
        path = f.name
    doc.saveas(path)
    return path


def _build_doc_with_only_hidden_titleblock():
    """合成DXF: 唯一のタイトルブロック（図面枠を含むINSERT）がoff+frozenの
    専用レイヤーに置かれている図面（実データ EE5322-455-02A.dxf/-18A.dxf を
    模したもの）。タイトルブロック外に可視ラベル'R10'を1つ配置する。"""
    doc = ezdxf.new()
    msp = doc.modelspace()
    _add_titleblock_block(doc, 'TITLEBLOCK')

    hidden_layer_name = 'OLD_TITLEBLOCK_LAYER'
    doc.layers.add(hidden_layer_name)
    hidden_layer = doc.layers.get(hidden_layer_name)
    hidden_layer.off()
    hidden_layer.freeze()

    # INSERT自身をoff+frozenレイヤーに置く（実データと同じ形。中身の個々の
    # エンティティのレイヤーは問わない——is_invisible()はINSERT自身の
    # チェックで中身ごと丸ごと除外する）。
    msp.add_blockref('TITLEBLOCK', (0, 0), dxfattribs={'layer': hidden_layer_name})
    msp.add_text('R10', dxfattribs={'insert': (10, 10)})
    return doc


def _build_doc_with_visible_and_hidden_titleblock():
    """合成DXF: 可視のタイトルブロック（図番 EE1111-001-01A）1つと、別の
    座標に置かれた非表示のタイトルブロック（図番 EE2222-002-02B）1つを持つ
    図面。表示中の図面枠が既に見つかるため、フォールバックは発動せず、
    非表示側の図番・タイトルは一切採用されないことを確認する（実データ
    `EE5322-470-38A.dxf` のパターン。2026-09-16修正の対象そのもの）。"""
    doc = ezdxf.new()
    msp = doc.modelspace()

    visible_block = doc.blocks.new(name='VISIBLE_TB')
    for p1, p2 in [((0, 0), (200, 0)), ((200, 0), (200, 100)),
                   ((200, 100), (0, 100)), ((0, 100), (0, 0))]:
        visible_block.add_line(p1, p2, dxfattribs={'lineweight': 100, 'color': 7})
    visible_block.add_text('TITLE', dxfattribs={'insert': (100, 50)})
    visible_block.add_text('VISIBLE TITLE', dxfattribs={'insert': (140, 50)})
    visible_block.add_text('EE1111-001-01A', dxfattribs={'insert': (170, 25)})
    msp.add_blockref('VISIBLE_TB', (0, 0))

    _add_titleblock_block(doc, 'HIDDEN_TB')
    hidden_layer_name = 'OLD_TITLEBLOCK_LAYER'
    doc.layers.add(hidden_layer_name)
    hidden_layer = doc.layers.get(hidden_layer_name)
    hidden_layer.off()
    hidden_layer.freeze()
    # 別座標に配置（1000,0 オフセット）。図番は _add_titleblock_block の
    # 'EE9999-001-01A' をそのまま使う（可視側'EE1111-001-01A'と区別できる）。
    msp.add_blockref('HIDDEN_TB', (1000, 0), dxfattribs={'layer': hidden_layer_name})
    return doc


# ---------------------------------------------------------------------------
# extract_labels() — 図番・タイトルの抽出、出力ラベルは表示中のみ
# ---------------------------------------------------------------------------

def test_extract_labels_recovers_title_and_drawing_number_via_fallback():
    """唯一のタイトルブロックが非表示レイヤーでも、フォールバックにより
    図番・タイトルが抽出できる。"""
    path = _save_and_load(_build_doc_with_only_hidden_titleblock())
    try:
        labels, info = extract_labels(
            path, extract_drawing_numbers_option=True, extract_title_option=True,
            original_filename='hidden_tb.dxf')
    finally:
        os.remove(path)
    assert info['main_drawing_number'] == 'EE9999-001-01A'
    assert info['title'] == 'SAMPLE TITLE'


def test_extract_labels_output_excludes_hidden_titleblock_text():
    """フォールバックは図番・タイトル判定にのみ使い、出力ラベル自体は
    常に表示中のエンティティのみに限定する（非表示タイトルブロックの
    'TITLE'・'DWG.No'・'SAMPLE TITLE'等は出力ラベルに混入しない）。"""
    path = _save_and_load(_build_doc_with_only_hidden_titleblock())
    try:
        labels, info = extract_labels(
            path, extract_drawing_numbers_option=True, extract_title_option=True,
            original_filename='hidden_tb.dxf')
    finally:
        os.remove(path)
    assert 'R10' in labels
    for leaked in ('TITLE', 'DWG.No', 'DATE', 'NAME', 'SAMPLE TITLE', 'EE9999-001-01A'):
        assert leaked not in labels


def test_extract_labels_visible_titleblock_takes_precedence_over_hidden():
    """表示中のタイトルブロックが既にある場合、フォールバックは発動せず、
    非表示側の図番・タイトルは一切採用されない（2026-09-16修正の対象を
    壊していないことの確認）。"""
    path = _save_and_load(_build_doc_with_visible_and_hidden_titleblock())
    try:
        labels, info = extract_labels(
            path, extract_drawing_numbers_option=True, extract_title_option=True,
            original_filename='visible_and_hidden.dxf')
    finally:
        os.remove(path)
    assert info['main_drawing_number'] == 'EE1111-001-01A'
    assert info['title'] == 'VISIBLE TITLE'
    assert 'EE9999-001-01A' not in labels
    assert 'EE2222-002-02B' not in labels


# ---------------------------------------------------------------------------
# ref_designator.collect_in_frame_labels() — 図面枠・情報枠の検出
# ---------------------------------------------------------------------------

def test_collect_in_frame_labels_detects_frame_via_fallback():
    """唯一のタイトルブロックが非表示レイヤーでも、フォールバックにより
    図面枠が検出され、warning（frame not found）が出ない。出力ラベルは
    表示中の'R10'のみで、非表示タイトルブロックの文字は混入しない。"""
    path = _save_and_load(_build_doc_with_only_hidden_titleblock())
    try:
        result = ref_designator.collect_in_frame_labels(path, frame_lineweight=100)
    finally:
        os.remove(path)
    assert result['error'] is None
    assert len(result['frames']) == 1
    texts = {t for (t, _x, _y) in result['labels']}
    assert 'R10' in texts
    for leaked in ('TITLE', 'DWG.No', 'DATE', 'NAME', 'SAMPLE TITLE', 'EE9999-001-01A'):
        assert leaked not in texts


def test_extract_ref_designator_data_no_warning_via_fallback():
    """`extract_ref_designator_data()`（通常モードの機器符号抽出）でも
    警告なしで図面枠内ラベルが収集できる。"""
    path = _save_and_load(_build_doc_with_only_hidden_titleblock())
    try:
        data = ref_designator.extract_ref_designator_data(
            path, frame_lineweight=100, original_filename='hidden_tb.dxf')
    finally:
        os.remove(path)
    assert data['warning'] is None
    assert data['frames'] == 1
    texts = {t for (t, _x, _y) in data['labels']}
    assert 'R10' in texts
    assert 'SAMPLE TITLE' not in texts


# ---------------------------------------------------------------------------
# region_detector.analyze_dxf_regions() — 図面枠の検出（領域検出の前提）
# ---------------------------------------------------------------------------

def test_analyze_dxf_regions_detects_frame_via_fallback():
    """唯一のタイトルブロックが非表示レイヤーでも、フォールバックにより
    図面枠が検出され、「領域探索を実施することができませんでした」の
    エラーにならない。"""
    path = _save_and_load(_build_doc_with_only_hidden_titleblock())
    try:
        result = analyze_dxf_regions(path)
    finally:
        os.remove(path)
    assert result['error'] is None
    assert len(result['frames']) == 1


def test_analyze_dxf_regions_visible_frame_unaffected_by_fallback():
    """表示中の図面枠が既にある通常の図面では、フォールバックが発動せず
    従来通りの結果になる（回帰なしの確認）。"""
    path = _save_and_load(_build_doc_with_visible_and_hidden_titleblock())
    try:
        result = analyze_dxf_regions(path)
    finally:
        os.remove(path)
    assert result['error'] is None
    # 可視側の図面枠(0,0)-(200,100)のみが検出される（非表示側(1000,*)は
    # 一切候補にならない）。
    assert result['frames'] == [(0.0, 200.0, 0.0, 100.0)]
