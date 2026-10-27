"""小さい閉じた箱（LWPOLYLINE）を境界線の辺にせず単独の候補面にすること、斜辺の補完面を作らないこと（2026-10-27）。

受入条件（ユーザー決定）:
  - 閉じた箱の LWPOLYLINE は単独で領域候補になり得るが、箱が結合した図形（CNJ1 部品の外形や、箱の辺を共有する面）は
    領域候補にしない。LWPOLYLINE を辺に分解するのは、閉じた長方形のうち面積が `area_ratio` 以上のものと、開いた LWPOLYLINE。
  - 元図面に無い斜めの辺（補完面の切り出しが始点へ直線で戻る辺）を持つ領域は作らない。
実データ: `EE6888-631-01A.dxf`（CNJ1 の24個の箱と縦棒）。
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from model.region_detector import _has_diagonal_edge, _resolve_complement_faces, analyze_dxf_regions  # noqa: E402

APP_CFG = {'detect_minor_regions': True, 'local_rotation': True}


def _find_sample(name):
    for root, _dirs, files in os.walk(os.path.join(PROJECT_ROOT, 'sample-dxf')):
        if name in files:
            return os.path.join(root, name)
    return None


def test_has_diagonal_edge():
    assert not _has_diagonal_edge([(0, 0), (10, 0), (10, 5), (0, 5)])
    assert _has_diagonal_edge([(81, 98), (81, 106), (41, 106), (41, 394)])   # 終端から始点へ戻る辺が斜め
    assert not _has_diagonal_edge([(0, 0), (10, 0), (10, 0.4), (0, 0.3)])   # 座標のずれ程度は斜辺としない


def _region(rid, polygon):
    from model.region_detector import _polygon_area, _polygon_corners
    return {'id': rid, 'frame': 0, 'polygon': polygon, 'corners': _polygon_corners(polygon),
            'area': _polygon_area(polygon), 'area_pct': 1.0, 'name_candidates': [('1.0', 'AAA')],
            'default_name': 'AAA', 'default_name_tier': 1, 'dangling_edges': [], '_tier_by_text': {}}


def test_complement_face_with_diagonal_closing_edge_is_dropped():
    """補完面（outline）から基準面（bar）を除いた残りの始点と終端が同じ辺上に無いとき、斜辺の領域を作らない。"""
    outline = [(81, 98), (81, 106), (41, 106), (41, 394.2), (81, 394.5), (90, 394.2), (90, 98)]
    bar = [(81, 98), (90, 98), (90, 394.5), (81, 394.5)]
    out = _resolve_complement_faces([_region(0, outline), _region(1, bar)], 100000.0)
    assert out and all(not _has_diagonal_edge(r['polygon']) for r in out)


@pytest.mark.slow
@pytest.mark.parametrize('name, gone', [
    ('EE6888-631-01A.dxf', {'SNM-7 DO7 LIFTER HI DN', 'SNM-7 DO15 SPARE'}),
    ('EE6892-631-01B.dxf', {'SNM-7 DO7 LIFTER HI DN', 'SNM-7 DO15 SPARE'}),
    ('EE6492-631-02A.dxf', {'SNM-7 DO7 LIFTER HI DN', 'SNM-7 DO15 SPARE'}),
    ('DE5027-563-03A.dxf', set()),     # 部品は LINE で描かれている。斜辺の領域だけが消える
])
def test_real_data_connector_boxes_do_not_form_regions(name, gone):
    path = _find_sample(name)
    assert path, f'{name} が sample-dxf に無い'
    a = analyze_dxf_regions(path, APP_CFG)
    every = a['regions'] + a['minor_regions']
    assert not [r['id'] for r in every if _has_diagonal_edge(r['polygon'])]
    names = {r['default_name'] for r in every if r['area_pct'] < 7}
    assert not (gone & names)
    # CNJ1 の箱24個＋縦棒が結合した外形（6.7%・30頂点）は領域にしない（LWPOLYLINE の図面）
    if gone:
        assert not [r for r in every if len(r['polygon']) >= 30 and round(r['area_pct'], 1) == 6.7]
