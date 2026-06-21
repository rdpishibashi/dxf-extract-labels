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
DANGLING = os.path.join(PROJECT_ROOT, 'EE6313-546-01E.dxf')  # 行き止まり枝(handle 214F/2199)を含む


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


@pytest.mark.skipif(not os.path.exists(MULTI) and not os.path.exists(SINGLE)
                     and not os.path.exists(ROTATED) and not os.path.exists(DANGLING),
                     reason='サンプル DXF が無い')
def test_no_region_has_consecutive_duplicate_corners():
    """`corners` に同じ座標が2回連続するエントリが無いこと（全サンプル共通の不変条件）。

    閉領域に寄与しない行き止まり枝（次数1のノードに繋がる境界線分）があると、
    半面探索がその枝を折り返すために生のポリゴンに「同じ頂点が2回連続する」
    アーティファクトを生んでいた（v1.5.7 で2-core抽出により修正）。"""
    for path in (MULTI, SINGLE, ROTATED, DANGLING):
        if not os.path.exists(path):
            continue
        a = analyze_dxf_regions(path)
        assert a['error'] is None
        for r in a['regions']:
            corners = r['corners']
            for i in range(len(corners)):
                prev = corners[i - 1]
                cur = corners[i]
                assert prev != cur, (
                    f"{os.path.basename(path)} region id={r['id']} has consecutive "
                    f"duplicate corner {cur}")


@pytest.mark.skipif(not os.path.exists(DANGLING), reason='サンプル DXF が無い')
def test_dangling_edges_reported_with_handles():
    """行き止まり枝(handle 214F: (660.53,129.56)-(807.24,129.56)、
    handle 2199: (812.24,235.81)-(812.24,129.56))が、その取り付け点を境界に持つ
    領域（図面1/領域1＝id 0）の `dangling_edges` に handle・座標付きで報告される
    こと（ユーザー報告: 頂点座標リストに (660.53, 129.56) が2回連続して現れる
    不具合の原因）。"""
    a = analyze_dxf_regions(DANGLING)
    assert a['error'] is None
    region0 = next(r for r in a['regions'] if r['id'] == 0)
    all_handles = {
        ent['handle']
        for br in region0['dangling_edges']
        for ent in br['entities']
    }
    assert '214F' in all_handles
    assert '2199' in all_handles


def test_dangling_edges_scoped_to_owning_region_only():
    """行き止まり枝は、その取り付け点(attachment)が境界に乗る領域だけに紐づき、
    無関係な部品・他領域のものは混在しない（ユーザー指摘: ファイル全体の158件
    ではなく、図面1/領域1に関係する2枝のみであるべき）。

    図面1/領域1(id 0)は #214F の単独枝と、#2199-#21AD-#219B-#21AA-#219F-#21A7
    が1本の枝（連結成分）にまとまった計2枝を持つ。#21A7 は領域0の上端境界の
    一部（(150.22,567.94)-(660.53,567.94)）としても使われている1本のLINEで、
    その延長（(660.53,567.94)-(812.24,567.94)）が行き止まりになっているため、
    同じ枝の構成エンティティとして含まれる。"""
    a = analyze_dxf_regions(DANGLING)
    assert a['error'] is None
    region0 = next(r for r in a['regions'] if r['id'] == 0)
    assert len(region0['dangling_edges']) == 2

    branch_handle_sets = [
        frozenset(ent['handle'] for ent in br['entities'])
        for br in region0['dangling_edges']
    ]
    assert frozenset(['214F']) in branch_handle_sets
    assert frozenset(['2199', '21AD', '219B', '21AA', '219F', '21A7']) in branch_handle_sets

    # 他の領域には(この図面では)行き止まり枝は無い。
    for r in a['regions']:
        if r['id'] != 0:
            assert r['dangling_edges'] == []


@pytest.mark.skipif(not os.path.exists(DANGLING), reason='サンプル DXF が無い')
def test_dangling_edge_pruning_dedupes_inner_outer_region_pair():
    """行き止まり枝の除去前は、同一の物理境界が「綺麗な内側面」と「行き止まり枝の
    往復で座標が汚れた外側面」の2つの異なる bbox を持つ別領域として重複検出されて
    いた（座標の汚れが bbox を変えるため、既存の bbox 重複除外をすり抜けていた）。
    除去後は同一 bbox になり正しく1領域に統合される（修正前は frames=1,regions=6
    で面積63.6%の領域が2件、修正後は5件で1件のみ。さらに v1.5.10 でPHANTOM線種の
    領域〈面積4.6%、後述 test_phantom_linetype_excluded_from_region_boundaries 参照〉
    を境界探索から除外したため、現在は4件）。"""
    a = analyze_dxf_regions(DANGLING)
    assert a['error'] is None
    assert len(a['regions']) == 4
    area_pcts = sorted(round(r['area_pct'], 1) for r in a['regions'])
    assert area_pcts == [1.8, 9.8, 52.6, 63.6]


@pytest.mark.skipif(not os.path.exists(DANGLING), reason='サンプル DXF が無い')
def test_phantom_linetype_excluded_from_region_boundaries():
    """境界線条件（lineweight=25/color=2）を満たしていても、線種(linetype)が
    PHANTOM（二点鎖線）の線は領域境界の探索対象から除外する（v1.5.10）。

    `EE6313-546-01E.dxf` には、実体の小さな矩形（handle 21AB/21AC/219A/219E、
    linetype=Continuous、面積1.8%、'MX CHAMBER'）の周囲に、別の handle
    （21AE/21A1/21A9/2198等、linetype=PHANTOM）で描かれた二点鎖線の矩形が重なって
    存在する。修正前はこのPHANTOM線も境界線として誤認識し、本来存在しない
    「実体矩形をくり抜いた」10角形・面積4.6%の領域が誤検出されていた
    （ユーザー報告: DXF-viewerで座標を確認したところ、抽出された境界の一部が
    実体の直線ではなく二点鎖線〈#21AB等〉だった）。除去後は実体の矩形
    （面積1.8%）のみが残り、PHANTOM由来の誤検出領域は消える。"""
    a = analyze_dxf_regions(DANGLING)
    assert a['error'] is None
    area_pcts = sorted(round(r['area_pct'], 1) for r in a['regions'])
    assert 4.6 not in area_pcts
    small_box = next(r for r in a['regions'] if round(r['area_pct'], 1) == 1.8)
    assert small_box['default_name'] == 'MX CHAMBER'
    assert small_box['corners'] == [
        (698.61, 238.2), (812.24, 238.2), (812.24, 309.1), (698.61, 309.1)]


@pytest.mark.skipif(not os.path.exists(DANGLING), reason='サンプル DXF が無い')
def test_nested_regions_each_get_own_confident_default_name():
    """入れ子/隣接する2領域が互いの候補リストに相手の名称を含む場合でも、
    各領域は自分自身の下端最近傍（Tier1）候補をデフォルト名称にする
    （ユーザー報告: 図面1/領域1,2 が同じ選択に同期され、ラベルはそれぞれ別の
    はずなのに一致してしまう不具合）。

    `EE6313-546-01E.dxf` の図面1/領域1(id 0, 面積63.6%)と領域2(id 1, 面積52.6%)
    は同名の2候補（B CHAMBER, BAKE HEATER UNIT RX）を共有するが、互いに
    下端最近傍の側が異なる（領域1→B CHAMBER、領域2→BAKE HEATER UNIT RX）。
    `default_name_tier` が両領域とも1（Tier1=下端最近傍）であることを確認し、
    `app.py` の他領域への選択同期がこのケースでは発動しない（=確信度の高い
    自前の候補を上書きしない）前提条件を保証する。"""
    a = analyze_dxf_regions(DANGLING)
    assert a['error'] is None
    region0 = next(r for r in a['regions'] if r['id'] == 0)
    region1 = next(r for r in a['regions'] if r['id'] == 1)
    assert region0['default_name'] == 'B CHAMBER'
    assert region0['default_name_tier'] == 1
    assert region1['default_name'] == 'BAKE HEATER UNIT RX'
    assert region1['default_name_tier'] == 1
    # 互いの候補リストに相手の名称が(優先度の低い候補として)含まれている
    # ことを確認する（同期バグが再発し得る前提条件そのものを保証する）。
    assert 'BAKE HEATER UNIT RX' in [t for _d, t in region0['name_candidates']]
    assert 'B CHAMBER' in [t for _d, t in region1['name_candidates']]


@pytest.mark.skipif(not os.path.exists(ROTATED), reason='サンプル DXF が無い')
def test_rotated_drawing_name_tier_uses_right_then_left_edge():
    """90°回転図面（回転角+90°多数派、`DE5434-553-10B.dxf`）では、下端相当の
    優先順位(Tier1)は右端の縦エッジ、上端相当(Tier2)は左端の縦エッジになる
    （ユーザー確認による仕様）。少なくとも一部の領域で Tier1/Tier2 の両方が
    実際に使われていることを確認する。"""
    a = analyze_dxf_regions(ROTATED)
    assert a['error'] is None
    tiers = {r['default_name_tier'] for r in a['regions'] if r['name_candidates']}
    assert 1 in tiers
    assert 2 in tiers
