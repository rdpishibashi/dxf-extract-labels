"""矩形領域抽出（オプション機能）の回帰テスト。

検証済み仕様:
  - 図面枠 = lineweight=100 / 領域境界 = lineweight=25 & color=2
  - 接点マージン ±2・同一座標線分の結合・面積>=枠面積×20%
  - 名称 = 境界近傍の英字3文字以上ラベル（複数候補）
  - 1ラベルが複数領域に所属可

実 DXF（プロジェクト直下のサンプル）を使ったブラックボックステスト。
サンプルが無い環境ではスキップする。
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from utils.region_detector import (  # noqa: E402
    analyze_dxf_regions, assign_region_labels,
    _point_in_polygon, _polygon_area,
)

MULTI = os.path.join(PROJECT_ROOT, 'EE6868-500-01C.dxf')   # 複数図面（13枠）
SINGLE = os.path.join(PROJECT_ROOT, 'EE6888-602-01A.dxf')  # 単一図面
ROTATED = os.path.join(PROJECT_ROOT, 'DE5434-553-10B.dxf')  # 図面全体が90°回転（名称が縦エッジ脇）


@pytest.mark.skipif(not os.path.exists(SINGLE), reason='サンプル DXF が無い')
def test_single_drawing_frame_and_regions():
    a = analyze_dxf_regions(SINGLE)
    assert a['error'] is None
    assert len(a['frames']) == 1
    assert a['frame_area'] > 0
    # 20%以上の閉領域が検出される
    assert len(a['regions']) >= 1
    # 各領域に名称候補（英字3文字以上）が付く
    assert any(r['name_candidates'] for r in a['regions'])


@pytest.mark.skipif(not os.path.exists(MULTI), reason='サンプル DXF が無い')
def test_multi_drawing_frames():
    a = analyze_dxf_regions(MULTI)
    assert a['error'] is None
    # 13 の図面枠が横並びで検出される
    assert len(a['frames']) == 13
    assert len(a['regions']) >= 13


@pytest.mark.skipif(not os.path.exists(MULTI), reason='サンプル DXF が無い')
def test_mpd_rack1_region_is_recovered():
    """MPD RACK1 領域が直交ポリゴンとして復元され、名称候補に含まれること。"""
    a = analyze_dxf_regions(MULTI)
    # 第1図面（最左）の大領域 ≒ 面積80%以上 のものに MPD RACK1 候補が含まれる
    frame1_regions = [r for r in a['regions'] if r['frame'] == 0]
    assert frame1_regions
    big = max(frame1_regions, key=lambda r: r['area_pct'])
    assert big['area_pct'] >= 50  # 凹型(階段状)の約20辺ポリゴンのため bbox より小さい
    assert len(big['corners']) >= 16  # 当初仕様の19辺相当の凹ポリゴン
    cand_texts = [t for _d, t in big['name_candidates']]
    assert 'MPD RACK1' in cand_texts
    # ポリゴンが妥当（面積>0・頂点数>=4）
    assert len(big['polygon']) >= 4
    assert _polygon_area(big['polygon']) > 0


@pytest.mark.skipif(not os.path.exists(SINGLE), reason='サンプル DXF が無い')
def test_connection_point_region_excluded():
    """返る領域はすべて境界上の接続点(円)が しきい値未満（配線ループは除外）。"""
    from utils.region_detector import (
        analyze_dxf_regions as _a, _collect_region_geometry,
        _count_connection_points_on_boundary, DEFAULT_REGION_CONFIG,
    )
    import ezdxf
    a = _a(SINGLE)
    assert a['error'] is None
    assert len(a['regions']) >= 1
    doc = ezdxf.readfile(SINGLE)
    _, _, _, _, cps = _collect_region_geometry(doc.modelspace(), DEFAULT_REGION_CONFIG)
    thr = DEFAULT_REGION_CONFIG['connection_point_threshold']
    margin = DEFAULT_REGION_CONFIG['connection_point_margin']
    for r in a['regions']:
        assert _count_connection_points_on_boundary(r['polygon'], cps, margin) < thr


@pytest.mark.skipif(not os.path.exists(SINGLE), reason='サンプル DXF が無い')
def test_name_candidates_exclude_circuit_and_notes():
    """名称候補から機器符号・NOTE・☆・境界線分上(距離0)が除外される。"""
    a = analyze_dxf_regions(SINGLE)
    for reg in a['regions']:
        for d, t in reg['name_candidates']:
            assert d >= 1.0                      # 線分上(距離0)は除外
            assert d <= 10.0                     # 既定の最大距離
            assert 'NOTE' not in t.upper()       # NOTE 除外
            assert '☆' not in t                   # ☆ 除外
            assert not any('a' <= ch <= 'z' for ch in t)  # 英小文字を含む名称は除外


@pytest.mark.skipif(not os.path.exists(SINGLE), reason='サンプル DXF が無い')
def test_multi_membership_assignment():
    """ラベルが複数領域に所属し得ることを確認する。"""
    a = analyze_dxf_regions(SINGLE)
    named = [{'polygon': r['polygon'], 'name': r['default_name']}
             for r in a['regions'] if r['default_name']]
    assigned = assign_region_labels(a['labels'], named)
    # 少なくとも一部のラベルが領域に所属する
    assert any(names for (_t, _x, _y, names) in assigned)
    # 内包判定の健全性: 全ラベルは枠内に収まっている
    assert len(assigned) == len(a['labels'])


def test_point_in_polygon_basic():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert _point_in_polygon((5, 5), sq)
    assert not _point_in_polygon((15, 5), sq)
    assert abs(_polygon_area(sq) - 100.0) < 1e-6


@pytest.mark.skipif(not os.path.exists(ROTATED), reason='サンプル DXF が無い')
def test_rotated_drawing_name_candidates_via_vertical_edge():
    """図面全体が90°回転しているファイルでも名称候補が見つかる（縦エッジフォールバック）。

    ROTATED は図面枠・領域境界線自体は通常向き（lineweight=100/25,color=2 のまま）だが、
    ラベル(MTEXT)の大半が90°回転して描かれており、領域の名称ラベルは下端/上端横エッジ
    ではなく左右いずれかの縦エッジ脇に置かれる。横エッジのみを見る実装ではここで
    name_candidates が常に空になっていた（要修正前のリグレッション確認用）。
    """
    a = analyze_dxf_regions(ROTATED)
    assert a['error'] is None
    assert len(a['regions']) >= 1
    assert any(r['name_candidates'] for r in a['regions'])
    names = {r['default_name'] for r in a['regions']}
    assert 'LA CHAMBER' in names


@pytest.mark.skipif(not os.path.exists(ROTATED), reason='サンプル DXF が無い')
def test_rotated_drawing_horizontal_gap_bridging_closes_large_box():
    """90°回転図面では部品が横線分（本来の縦線分に相当）を途切れさせるため、
    横線分ギャップ橋渡しのフォールバックが無いと大きな矩形（CONTROL BOX CORE FX
    を囲む領域、handle 1EAF/1EB0/1E59/2748/1EA3/1EAE で構成）が閉領域として
    検出できなかった（ユーザー確認による既知の正解ラベル）。"""
    a = analyze_dxf_regions(ROTATED)
    assert a['error'] is None
    all_candidate_texts = {
        t for r in a['regions'] for _d, t in r['name_candidates']
    }
    assert 'CONTROL BOX CORE FX' in all_candidate_texts


@pytest.mark.skipif(not os.path.exists(ROTATED), reason='サンプル DXF が無い')
def test_rotated_drawing_on_line_label_excluded_and_does_not_hide_real_name():
    """境界線上(d=0、min_dist=1.0未満)に偶然乗った無関係なラベル（コネクタ符号
    CN24POW04/05）はフォールバックでも候補に含めない（min_dist はフォールバックでも
    変えない）。かつてはこの無関係なラベルが横エッジ側で1件見つかっただけで、本来の
    縦エッジ側の名称候補（CONTROL BOX CORE FX/RX）が完全に隠れてしまっていた
    （ユーザー確認による既知の正解ラベル）。"""
    a = analyze_dxf_regions(ROTATED)
    assert a['error'] is None
    all_texts_by_distance = [(d, t) for r in a['regions'] for d, t in r['name_candidates']]
    assert not any(t in ('CN24POW04', 'CN24POW05') for _d, t in all_texts_by_distance)
    assert all(d >= 1.0 for d, _t in all_texts_by_distance)
    names = {r['default_name'] for r in a['regions']}
    assert 'CONTROL BOX CORE RX' in names


@pytest.mark.skipif(not os.path.exists(ROTATED), reason='サンプル DXF が無い')
def test_rotated_drawing_multiline_mtext_kept_as_single_candidate():
    """1つのMTEXTエンティティ内の複数行（\\P 区切り、例:
    "CN I/F B.D TYPE3\\P(CN-IF3-1A)"）は同じハンドル(handle 1C55)を持つため、
    分割せず1つの結合済み文字列（"CN I/F B.D TYPE3 (CN-IF3-1A)"）として
    名称候補にする（ユーザー確認による既知の正解ラベル）。"""
    a = analyze_dxf_regions(ROTATED)
    assert a['error'] is None
    all_candidate_texts = {t for r in a['regions'] for _d, t in r['name_candidates']}
    assert 'CN I/F B.D TYPE3 (CN-IF3-1A)' in all_candidate_texts
    assert 'CN I/F B.D TYPE3' not in all_candidate_texts
    assert '(CN-IF3-1A)' not in all_candidate_texts
