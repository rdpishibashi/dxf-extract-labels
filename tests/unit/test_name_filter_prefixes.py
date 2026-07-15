"""name_filter_prefixes（名称候補の前方一致による面積閾値バイパス採用、v1.9.3）の
合成DXFによる end-to-end 確認。

ユーザー要望: 抽出したい領域の名称が分かっている場合、面積比とは無関係にその
領域を抽出したい。`analyze_dxf_regions` の採用判定に、名称候補が
`name_filter_prefixes` のいずれかで前方一致する候補は面積閾値を問わず採用する
条件を追加した。

組み合わせ表:
  面積条件(満たす/満たさない) × 文字列指定(前方一致/不一致/未指定)
  + 複数prefix指定 + 全角/半角混在 + min_face_ratio下限は救済されないこと
  + 部分一致ではなく前方一致であること
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


def _add_rect(msp, x0, y0, x1, y1, lineweight, color):
    for p1, p2 in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                   ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
        msp.add_line(p1, p2, dxfattribs={'lineweight': lineweight, 'color': color})


def _build_doc(small_label, small_bbox=(10, 10, 30, 30), tiny_bbox=None, tiny_label=None):
    """外枠(0,0)-(200,100)、frame_area=20000 + 面積5%未満の独立矩形1つ（+任意でノイズ矩形）。"""
    doc = ezdxf.new()
    msp = doc.modelspace()
    _add_rect(msp, 0, 0, 200, 100, FRAME_LW, FRAME_COLOR)
    _add_rect(msp, 0, 0, 200, 100, BOUNDARY_LW, BOUNDARY_COLOR)
    x0, y0, x1, y1 = small_bbox
    _add_rect(msp, x0, y0, x1, y1, BOUNDARY_LW, BOUNDARY_COLOR)
    msp.add_text(small_label, dxfattribs={'insert': (x0 + 1, y1 + 1)})
    if tiny_bbox:
        tx0, ty0, tx1, ty1 = tiny_bbox
        _add_rect(msp, tx0, ty0, tx1, ty1, BOUNDARY_LW, BOUNDARY_COLOR)
        msp.add_text(tiny_label, dxfattribs={'insert': (tx0 + 0.5, ty1 + 0.5)})
    return doc


def _analyze(doc, cfg):
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as f:
        path = f.name
    doc.saveas(path)
    try:
        return analyze_dxf_regions(path, cfg)
    finally:
        os.remove(path)


def _names(a):
    return sorted(r['default_name'] for r in a['regions'])


# 小矩形(10,10)-(30,30) = 面積400 = frame_area(20000)比2% → area_ratio既定5%未満


def test_below_threshold_excluded_without_prefix_filter():
    """文字列指定なしなら、面積閾値未満の候補は従来通り除外される。"""
    doc = _build_doc('SMALL TARGET BOX')
    a = _analyze(doc, DEFAULT_REGION_CONFIG)
    assert a['error'] is None
    assert 'SMALL TARGET BOX' not in _names(a)


def test_below_threshold_included_with_matching_prefix():
    """面積閾値未満でも、名称候補が前方一致すれば強制採用される（新機能）。"""
    doc = _build_doc('SMALL TARGET BOX')
    cfg = dict(DEFAULT_REGION_CONFIG)
    cfg['name_filter_prefixes'] = ('SMALL',)
    a = _analyze(doc, cfg)
    assert a['error'] is None
    assert 'SMALL TARGET BOX' in _names(a)


def test_below_threshold_excluded_with_non_matching_prefix():
    """指定した文字列がどの候補にも前方一致しなければ、従来通り除外される。"""
    doc = _build_doc('SMALL TARGET BOX')
    cfg = dict(DEFAULT_REGION_CONFIG)
    cfg['name_filter_prefixes'] = ('ZZZ',)
    a = _analyze(doc, cfg)
    assert a['error'] is None
    assert 'SMALL TARGET BOX' not in _names(a)


def test_above_threshold_included_regardless_of_prefix_filter():
    """面積閾値を満たす候補は、文字列指定の有無に関わらず従来通り採用される
    （面積条件と文字列条件はOR、片方だけでも十分）。"""
    doc = ezdxf.new()
    msp = doc.modelspace()
    _add_rect(msp, 0, 0, 200, 100, FRAME_LW, FRAME_COLOR)
    _add_rect(msp, 0, 0, 200, 100, BOUNDARY_LW, BOUNDARY_COLOR)
    _add_rect(msp, 10, 10, 110, 90, BOUNDARY_LW, BOUNDARY_COLOR)  # 100*80=8000=40% >= 5%
    msp.add_text('BIG TARGET BOX', dxfattribs={'insert': (11, 91)})
    cfg = dict(DEFAULT_REGION_CONFIG)
    cfg['name_filter_prefixes'] = ('ZZZ',)  # 一致しない文字列を指定しても
    a = _analyze(doc, cfg)
    assert a['error'] is None
    assert 'BIG TARGET BOX' in _names(a)  # 面積条件だけで採用される


def test_multiple_prefixes_any_match_included():
    """複数の前方一致文字列を指定した場合、いずれか1つに一致すれば採用される。"""
    doc = _build_doc('SMALL TARGET BOX')
    cfg = dict(DEFAULT_REGION_CONFIG)
    cfg['name_filter_prefixes'] = ('ZZZ', 'YYY', 'SMALL')
    a = _analyze(doc, cfg)
    assert a['error'] is None
    assert 'SMALL TARGET BOX' in _names(a)


def test_zenkaku_label_matched_by_hankaku_prefix():
    """全角ラベルに対して半角の前方一致文字列を指定しても一致すること
    （normalize_width による半角化後の前方一致判定、4節の鉄則）。"""
    doc = _build_doc('ＳＭＡＬＬ ＴＡＲＧＥＴ')
    cfg = dict(DEFAULT_REGION_CONFIG)
    cfg['name_filter_prefixes'] = ('SMALL',)
    a = _analyze(doc, cfg)
    assert a['error'] is None
    assert 'ＳＭＡＬＬ ＴＡＲＧＥＴ' in _names(a)


def test_min_face_ratio_floor_not_rescued_by_prefix_filter():
    """min_face_ratio（ノイズ除去の絶対下限、既定0.5%）未満の候補は、前方一致
    しても救済されない（意図的なノイズ下限、面積閾値バイパスの対象外）。"""
    doc = ezdxf.new()
    msp = doc.modelspace()
    _add_rect(msp, 0, 0, 200, 100, FRAME_LW, FRAME_COLOR)
    _add_rect(msp, 0, 0, 200, 100, BOUNDARY_LW, BOUNDARY_COLOR)
    _add_rect(msp, 10, 10, 12, 14, BOUNDARY_LW, BOUNDARY_COLOR)  # 2*4=8=0.04%<0.5%
    msp.add_text('TINY TARGET BOX', dxfattribs={'insert': (10.2, 14.2)})
    cfg = dict(DEFAULT_REGION_CONFIG)
    cfg['name_filter_prefixes'] = ('TINY',)
    a = _analyze(doc, cfg)
    assert a['error'] is None
    assert 'TINY TARGET BOX' not in _names(a)


def test_prefix_match_is_prefix_not_substring():
    """`name_filter_prefixes` は前方一致であり部分一致ではないこと。ラベルの
    途中に含まれる文字列を指定しても一致しない。"""
    doc = _build_doc('SMALL TARGET BOX')
    cfg = dict(DEFAULT_REGION_CONFIG)
    cfg['name_filter_prefixes'] = ('TARGET',)  # 先頭ではなく途中の語
    a = _analyze(doc, cfg)
    assert a['error'] is None
    assert 'SMALL TARGET BOX' not in _names(a)


def test_empty_prefix_list_behaves_as_default():
    """`name_filter_prefixes` が空（既定値）なら、従来通りの面積閾値のみで
    判定される（後方互換）。"""
    doc = _build_doc('SMALL TARGET BOX')
    a = _analyze(doc, dict(DEFAULT_REGION_CONFIG))
    assert a['error'] is None
    assert 'SMALL TARGET BOX' not in _names(a)
