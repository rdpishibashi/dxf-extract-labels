"""このテストが守るもの: `model.region_decisions.classify_regions()` の6分類と判定の優先順位。

出力（`gather_name_selections`）・「領域の確認」の対象判定・viewer の表示が同じ分類を使うため、
ここで固定した優先順位が3者の一致の根拠になる。
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model import region_decisions as rd  # noqa: E402

SQ = [(0, 0), (10, 0), (10, 10), (0, 10)]


def _reg(rid, cands, frame=0, area=12.0):
    return {'id': rid, 'frame': frame, 'polygon': SQ, 'corners': SQ, 'area_pct': area,
            'name_candidates': list(cands)}


def _classify(regions, checked=(), excluded=(), adopted=lambda label: True):
    analysis = {'regions': regions, 'labels': [], 'main_drawing_number': 'X'}
    return rd.classify_regions(analysis, set(checked), set(excluded), adopted)


def test_six_statuses():
    regions = [
        _reg(0, [(1, 'EX'), (2, 'A')]),            # 除外（候補のどれかが除外）
        _reg(1, []),                               # 名称なし
        _reg(2, [(1, 'ONE'), (2, 'OFF')]),         # 確定（チェック済みがちょうど1件）
        _reg(3, [(1, 'P'), (2, 'Q')]),             # 要選択
        _reg(4, [(1, 'NOPE')]),                    # 特定OFF（チェック済み0件）
    ]
    d = _classify(regions, checked={'ONE', 'P', 'Q', 'A'}, excluded={'EX'})
    assert [d[i].status for i in range(5)] == [
        rd.EXCLUDED, rd.UNNAMED, rd.DECIDED, rd.PENDING, rd.UNSPECIFIED]
    assert d[2].name == 'ONE'
    assert d[3].candidates == ((1, 'P'), (2, 'Q'))


def test_excluded_wins_over_checked_candidate():
    """同じ名称が「特定」でも「除外」でも、除外が優先される。"""
    d = _classify([_reg(0, [(1, 'A')])], checked={'A'}, excluded={'A'})
    assert d[0].status == rd.EXCLUDED


def test_unnamed_region_exclusion_uses_display_label():
    """名称なし領域は「抽出領域の選択」の表示名で除外を判定する。除外されていなければ UNNAMED。"""
    analysis = {'regions': [_reg(0, [])], 'labels': [], 'main_drawing_number': 'X'}
    from model.ref_designator import unnamed_region_labels
    label = unnamed_region_labels(analysis)[0]
    assert rd.classify_regions(analysis, set(), {label})[0].status == rd.EXCLUDED
    assert rd.classify_regions(analysis, set(), set())[0].status == rd.UNNAMED


def test_unnamed_adopted_follows_callback():
    d_on = _classify([_reg(0, [])], adopted=lambda label: True)
    d_off = _classify([_reg(0, [])], adopted=lambda label: False)
    assert d_on[0].adopted is True and d_off[0].adopted is False


def test_pending_region_decided_by_other_frame():
    """別の図面枠で1件に決まった名称と一致する要選択の領域は OTHER_FRAME。同じ図面枠では根拠にしない。"""
    regions = [_reg(0, [(1, 'A'), (2, 'B')], frame=0), _reg(1, [(1, 'A')], frame=1)]
    d = _classify(regions, checked={'A', 'B'})
    assert d[1].status == rd.DECIDED
    assert (d[0].status, d[0].name) == (rd.OTHER_FRAME, 'A')

    same_frame = [_reg(0, [(1, 'A'), (2, 'B')], frame=0), _reg(1, [(1, 'A')], frame=0)]
    assert _classify(same_frame, checked={'A', 'B'})[0].status == rd.PENDING


def test_other_frame_ignores_excluded_region_as_evidence():
    regions = [_reg(0, [(1, 'A'), (2, 'B')], frame=0), _reg(1, [(1, 'A'), (2, 'EX')], frame=1)]
    d = _classify(regions, checked={'A', 'B', 'EX'}, excluded={'EX'})
    assert d[1].status == rd.EXCLUDED
    assert d[0].status == rd.PENDING
