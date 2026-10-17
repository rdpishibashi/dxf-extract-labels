"""図面情報枠（タイトルブロック）を図面枠右下の固定サイズで決める（2026-10-17、ユーザー確認）。

不具合: 境界線探索で位置を決めていたため、テンプレート内にタイトルブロック本来の左辺・上辺の線が
見つからない図面（YO26-0815-2 の13件）で左端が図面枠の左辺・上端が内部線（高さ82）になり、
図面下部の横幅全体が除外されて回路ラベル（`EE3273-039-06B.dxf` の `SENSOR I/F B.D-Ⅲ`・`OSMG`・`CN6`）が
出力から落ちていた。逆に内部線を上辺に取り違える図面（高さ52の19件）では改訂履歴
（`2016/02/15`・`I CAD変換/野村` 等）が出力に漏れていた。

保証: 位置は図面枠の右下に固定サイズ（新テンプレート 175×64、項目名が右175より左にもあれば
旧テンプレート 350×72）。「情報枠があるか」の判定（項目名の一致数）は従来どおり。
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from model import ref_designator as rd  # noqa: E402

XL, XR, Y0, Y1 = 25.0, 410.0, 10.0, 287.0
TERMS_NEW = [('MFG No.', 236.8, 15.8), ('DWG No.', 313.7, 15.8), ('DATE', 305.5, 24.5),
             ('SCALE', 305.5, 31.5), ('TITLE', 338.8, 43.9), ('APPRV', 238.4, 45.0)]


def test_new_template_is_fixed_175_by_64_even_when_inner_lines_mislead():
    """境界線が図面枠の左辺・y92 の線しか無くても固定サイズになる（06B の再現）。"""
    misleading_lines = [((XL, 92.0), (XR, 92.0)), ((XL, Y0), (XL, 92.0))]
    assert rd._detect_info_frame_bbox(misleading_lines, TERMS_NEW, XL, XR, Y0, Y1) == (XR - 175.0, XR, Y0, Y0 + 64.0)


def test_old_template_when_connected_terms_reach_left_of_the_new_width():
    """右辺から連なる項目名（間隔30以内）が右175より左へ及ぶ→旧テンプレート（実データの旧テンプレートと同じ並び）。"""
    terms = TERMS_NEW + [('CHECK', x, 45.0) for x in (280.0, 255.0, 215.0, 190.0, 165.0, 140.0, 115.0, 90.0, 65.0)]
    assert rd._detect_info_frame_bbox([], terms, XL, XR, Y0, Y1) == (XR - 350.0, XR, Y0, Y0 + 72.0)


def test_separate_title_block_on_the_left_does_not_make_it_old():
    """横並びの別タイトルブロック（間隔が30を超えて離れた項目名）は旧テンプレート判定に使わない。"""
    terms = TERMS_NEW + [('CHECK', 100.0, 45.0), ('DRAW', 150.0, 45.0), ('MARK', 120.0, 45.0)]
    assert rd._detect_info_frame_bbox([], terms, XL, XR, Y0, Y1) == (XR - 175.0, XR, Y0, Y0 + 64.0)


def test_terms_far_above_the_title_block_do_not_make_it_old():
    """履歴表など上のほうにある項目名（y > y0+74）は旧テンプレート判定に使わない。"""
    terms = TERMS_NEW + [('MARK', 100.0, 200.0)]
    assert rd._detect_info_frame_bbox([], terms, XL, XR, Y0, Y1) == (XR - 175.0, XR, Y0, Y0 + 64.0)


def test_width_never_exceeds_the_frame():
    terms = [('MFG No.', 40.0, 15.8), ('DATE', 60.0, 24.5), ('TITLE', 80.0, 43.9), ('CHECK', 30.0, 45.0)]
    left, xr, _y0, _top = rd._detect_info_frame_bbox([], terms, 25.0, 150.0, Y0, Y1)
    assert left == 25.0 and xr == 150.0


def test_fewer_than_three_terms_means_no_info_frame():
    assert rd._detect_info_frame_bbox([], TERMS_NEW[:2], XL, XR, Y0, Y1) is None


# ---------------------------------------------------------------------------
# 実データ
# ---------------------------------------------------------------------------

SAMPLE_DIR = os.path.join(PROJECT_ROOT, 'sample-dxf')


def _find_sample(name):
    for root, _dirs, files in os.walk(SAMPLE_DIR):
        if name in files:
            return os.path.join(root, name)
    return None


def _labels(name):
    path = _find_sample(name)
    if path is None:
        pytest.skip('サンプル DXF が無い')
    data = rd.extract_ref_designator_data(path, frame_lineweight=100, original_filename=name)
    return {t for t, _x, _y in data['labels']}


@pytest.mark.slow
def test_real_data_circuit_labels_below_the_title_block_are_extracted():
    labels = _labels('EE3273-039-06B.dxf')
    assert {'SENSOR I/F B.D-III', 'OSMG', 'CN6'} <= labels
    assert not ({'MFG No.', 'APPRV', 'DWG No.', 'SCALE'} & labels)   # タイトル項目名は出ない


@pytest.mark.slow
def test_real_data_revision_history_is_not_leaked():
    labels = _labels('EE2685-335-01D.dxf')   # 以前は上辺を高さ52に取り違えて改訂履歴が漏れた
    assert not ({'2016/02/15', 'I CAD変換/野村'} & labels)
