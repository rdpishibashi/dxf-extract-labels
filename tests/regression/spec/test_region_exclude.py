"""領域名ごとの「特定／除外」指定（v3.11.0）の仕様テスト。

**守る対象**: 領域名の名称候補が「除外」指定された場合、その領域内（境界線上を
含む）のラベルが抽出結果から完全に取り除かれること。単体テストだけでは
「`excluded_names`/`master_index` の渡し漏れでその経路だけ黙って除外が効かない」
という不具合を検出できないため（`Tools/CLAUDE.md`に記録された
`extract_title_option`/`extract_drawing_numbers_option`の実際の事故と同じ
クラス）、`model/ref_designator.py`の純粋関数から`model/excel_output.py`の
出力関数まで、実際にExcelを生成してopenpyxlで読み戻し確認する。

2026-10-08、引き継ぎ書（§9・§11）に基づく。
"""
import os
import sys
from io import BytesIO

import openpyxl
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from model import ref_designator  # noqa: E402
from model.excel_output import create_region_excel_output  # noqa: E402


def _load_wb(data: bytes):
    return openpyxl.load_workbook(BytesIO(data))


# ---------------------------------------------------------------------------
# excluded_region_ids — 候補数×除外指定の組み合わせ（表2）
# ---------------------------------------------------------------------------

def _analysis(regions):
    return {'regions': regions}


def test_excluded_region_ids_single_candidate_match():
    """候補1件のみの領域で、その名称が除外指定なら除外される。"""
    analysis = _analysis([
        {'id': 0, 'name_candidates': [(1.0, 'AWG20')]},
    ])
    assert ref_designator.excluded_region_ids(analysis, {'AWG20'}) == {0}


def test_excluded_region_ids_no_match():
    """除外指定された名称がどの候補にも一致しなければ除外されない。"""
    analysis = _analysis([
        {'id': 0, 'name_candidates': [(1.0, 'RACK1')]},
    ])
    assert ref_designator.excluded_region_ids(analysis, {'AWG20'}) == set()


def test_excluded_region_ids_multiple_candidates_one_matches():
    """候補2件以上のうち1件だけが除外指定でも、その領域全体が除外される
    （どの名前を採用するかに関わらず、2026-10-08ユーザー決定）。"""
    analysis = _analysis([
        {'id': 0, 'name_candidates': [(1.0, 'RACK1'), (2.0, 'AWG20')]},
    ])
    assert ref_designator.excluded_region_ids(analysis, {'AWG20'}) == {0}


def test_excluded_region_ids_no_candidates_never_excluded():
    """候補0件（無名領域）は指定する名称自体が存在しないため、除外されない
    （excluded_names に何が入っていても関係ない。2026-10-08ユーザー承認）。"""
    analysis = _analysis([
        {'id': 0, 'name_candidates': []},
    ])
    assert ref_designator.excluded_region_ids(analysis, {'AWG20', 'RACK1'}) == set()


def test_excluded_region_ids_normalizes_width():
    """候補テキストの全角/半角表記の揺れを正規化してから比較する。"""
    analysis = _analysis([
        {'id': 0, 'name_candidates': [(1.0, 'ＡＷＧ２０')]},
    ])
    assert ref_designator.excluded_region_ids(analysis, {'AWG20'}) == {0}


def test_excluded_region_ids_empty_excluded_names_returns_empty():
    """除外指定が1件も無ければ、候補の有無に関わらず空集合を返す。"""
    analysis = _analysis([
        {'id': 0, 'name_candidates': [(1.0, 'RACK1')]},
    ])
    assert ref_designator.excluded_region_ids(analysis, set()) == set()


def test_excluded_region_ids_multiple_regions_independent():
    """複数領域がある場合、一致した領域のみが集合に入る。"""
    analysis = _analysis([
        {'id': 0, 'name_candidates': [(1.0, 'RACK1')]},
        {'id': 1, 'name_candidates': [(1.0, 'AWG20')]},
        {'id': 2, 'name_candidates': [(1.0, 'AWG24')]},
    ])
    assert ref_designator.excluded_region_ids(analysis, {'AWG20', 'AWG24'}) == {1, 2}


# ---------------------------------------------------------------------------
# filter_labels_outside_excluded — ラベル位置×除外領域（表3）
# ---------------------------------------------------------------------------

_SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_filter_removes_label_inside_excluded_polygon():
    """除外領域の内側にあるラベルは消える。"""
    labels = [('INSIDE', 5.0, 5.0)]
    kept = ref_designator.filter_labels_outside_excluded(labels, [_SQUARE])
    assert kept == []


def test_filter_removes_label_on_boundary():
    """除外領域の境界線上（boundary_eps以内）のラベルも消える
    （要求「領域線上も含めて」。既存の_point_in_polygonの挙動を使う）。"""
    labels = [('ON_EDGE', 0.0, 5.0)]
    kept = ref_designator.filter_labels_outside_excluded(labels, [_SQUARE])
    assert kept == []


def test_filter_keeps_label_outside_excluded_polygon():
    """除外領域の外側にあるラベルは残る。"""
    labels = [('OUTSIDE', 50.0, 50.0)]
    kept = ref_designator.filter_labels_outside_excluded(labels, [_SQUARE])
    assert kept == [('OUTSIDE', 50.0, 50.0)]


def test_filter_removes_label_in_overlap_with_specified_region():
    """特定領域と除外領域が重なる部分のラベルも消える（除外優先）。
    filter_labels_outside_excluded() 自体は「特定」領域を知らないので、
    重なりの有無に関わらず除外ポリゴン内のラベルを一律に消す（除外優先は
    呼び出し元がbuild_region_output()に渡すラベル集合を絞り込む形で実現する
    ため、この関数のレベルでは「除外ポリゴンに入っているか」だけを見る）。"""
    overlapping_square = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
    labels = [('IN_BOTH', 7.0, 7.0)]
    kept = ref_designator.filter_labels_outside_excluded(labels, [_SQUARE, overlapping_square])
    assert kept == []


def test_filter_removes_label_in_nested_region():
    """入れ子（除外領域の内側に別の特定領域）でも、外側の除外ポリゴンに
    内包される時点でラベルは消える（2026-10-08ユーザー承認）。"""
    inner_square = [(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)]
    labels = [('IN_INNER', 5.0, 5.0)]
    # 外側の除外ポリゴンのみを渡しても、内側のラベルは消える
    kept = ref_designator.filter_labels_outside_excluded(labels, [_SQUARE])
    assert kept == []
    # 内側ポリゴン自体が除外対象でなくても、外側が除外ならラベルは消える
    assert inner_square  # 内側ポリゴンの定義自体はここでは使わない（上記で確認済み）


def test_filter_anchor_outside_text_overflowing_is_kept():
    """アンカー点が領域外なら、テキストが領域にはみ出していても残る
    （既存の`領域`列割り当てと同じ、アンカー点のみで判定する規約。
    2026-10-08、意図的な既知の制限として固定）。"""
    labels = [('ANCHOR_OUTSIDE', 10.5, 5.0)]  # 境界(x=10)からわずかに外
    kept = ref_designator.filter_labels_outside_excluded(labels, [_SQUARE])
    assert kept == [('ANCHOR_OUTSIDE', 10.5, 5.0)]


def test_filter_no_excluded_polygons_returns_all_labels():
    """除外ポリゴンが0件なら、全ラベルがそのまま残る（回帰防護）。"""
    labels = [('A', 1.0, 1.0), ('B', 2.0, 2.0)]
    assert ref_designator.filter_labels_outside_excluded(labels, []) == labels


def test_filter_preserves_label_order_and_duplicates():
    """複数ラベルが残る場合、順序と重複はそのまま保持される。"""
    labels = [('A', 50.0, 50.0), ('A', 60.0, 60.0), ('B', 5.0, 5.0)]
    kept = ref_designator.filter_labels_outside_excluded(labels, [_SQUARE])
    assert kept == [('A', 50.0, 50.0), ('A', 60.0, 60.0)]


# ---------------------------------------------------------------------------
# build_all_regions_summary — 'mark' の付与（除外 > 特定 > 空欄）
# ---------------------------------------------------------------------------

_SQUARE_100 = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def test_build_all_regions_summary_mark_excluded():
    """excluded_ids に含まれる領域は mark='除外'。"""
    analysis = {'regions': [
        {'id': 0, 'frame': 0, 'polygon': _SQUARE_100, 'area_pct': 50.0,
         'name_candidates': [(1.0, 'AWG20')]},
    ]}
    rows = ref_designator.build_all_regions_summary(
        analysis, {}, 'a.dxf', [], excluded_ids={0}, named_ids=set())
    assert rows[0]['mark'] == '除外'


def test_build_all_regions_summary_mark_named():
    """named_ids に含まれる（除外ではない）領域は mark='特定'。"""
    analysis = {'regions': [
        {'id': 0, 'frame': 0, 'polygon': _SQUARE_100, 'area_pct': 50.0,
         'name_candidates': [(1.0, 'RACK1')]},
    ]}
    rows = ref_designator.build_all_regions_summary(
        analysis, {('a.dxf', 0): ['RACK1']}, 'a.dxf', [],
        excluded_ids=set(), named_ids={0})
    assert rows[0]['mark'] == '特定'


def test_build_all_regions_summary_mark_empty_when_neither():
    """除外でも特定でもない領域（候補はあるが未指定）は mark=''。"""
    analysis = {'regions': [
        {'id': 0, 'frame': 0, 'polygon': _SQUARE_100, 'area_pct': 50.0,
         'name_candidates': [(1.0, 'RACK1')]},
    ]}
    rows = ref_designator.build_all_regions_summary(
        analysis, {}, 'a.dxf', [], excluded_ids=set(), named_ids=set())
    assert rows[0]['mark'] == ''


def test_build_all_regions_summary_mark_excluded_takes_priority_over_named():
    """同じ領域idが excluded_ids と named_ids の両方に含まれる場合、
    '除外' が優先される（除外優先、2026-10-08ユーザー決定）。"""
    analysis = {'regions': [
        {'id': 0, 'frame': 0, 'polygon': _SQUARE_100, 'area_pct': 50.0,
         'name_candidates': [(1.0, 'RACK1')]},
    ]}
    rows = ref_designator.build_all_regions_summary(
        analysis, {('a.dxf', 0): ['RACK1']}, 'a.dxf', [],
        excluded_ids={0}, named_ids={0})
    assert rows[0]['mark'] == '除外'


def test_build_all_regions_summary_label_count_uses_pre_exclusion_labels():
    """領域内ラベル数は除外前の件数（呼び出し元が除外前のlabelsを渡す前提、
    2026-10-08ユーザー決定）。この関数自体はexcluded_idsをラベル絞り込みには
    使わない——集計対象のlabels次第であることを確認する。"""
    analysis = {'regions': [
        {'id': 0, 'frame': 0, 'polygon': _SQUARE_100, 'area_pct': 50.0,
         'name_candidates': [(1.0, 'AWG20')]},
    ]}
    labels = [('L1', 10.0, 10.0), ('L2', 20.0, 20.0)]
    rows = ref_designator.build_all_regions_summary(
        analysis, {}, 'a.dxf', labels, excluded_ids={0}, named_ids=set())
    assert rows[0]['label_count'] == 2   # 除外前の件数のまま


# ---------------------------------------------------------------------------
# build_ref_designator_region_results(..., excluded_names=...) — 統合検証
# ---------------------------------------------------------------------------

def _region_results_fixture(excluded_names=frozenset()):
    """AWG20(BL)領域(ラベルCB001)とRACK1領域(ラベルU1)を持つ合成データ。
    2領域は重ならない。"""
    square_a = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    square_b = [(50.0, 50.0), (60.0, 50.0), (60.0, 60.0), (50.0, 60.0)]
    ref_data_by_file = {
        'a.dxf': {
            'labels': [('CB001', 5.0, 5.0), ('U1', 55.0, 55.0)],
            'total_in_frame': 2,
        },
    }
    region_analyses = {
        'a.dxf': {
            'frames': [(0, 0, 0, 0)],
            'regions': [
                {'id': 0, 'frame': 0, 'polygon': square_a, 'area_pct': 10.0,
                 'name_candidates': [(1.0, 'AWG20(BL)')]},
                {'id': 1, 'frame': 0, 'polygon': square_b, 'area_pct': 10.0,
                 'name_candidates': [(1.0, 'RACK1')]},
            ],
            'main_drawing_number': 'EE0001',
            'title': None, 'subtitle': None,
        },
    }
    name_selections_by_file = {'a.dxf': {('a.dxf', 1): ['RACK1']}}
    return ref_designator.build_ref_designator_region_results(
        ref_data_by_file, region_analyses, name_selections_by_file,
        sort_value='asc', excluded_names=excluded_names)


def test_region_results_no_exclusion_matches_legacy_output():
    """excluded_names が空のとき、従来と完全に同一の出力になる（回帰防護）。"""
    r = _region_results_fixture()['a.dxf']
    assert r['total_in_frame'] == 2
    labels_in_rows = {row['ラベル'] for row in r['rows']}
    assert labels_in_rows == {'CB001', 'U1'}
    assert r['ref_designator_count'] == 2  # 両方CAN


def test_region_results_excluded_region_removes_its_labels():
    """AWG20(BL)を除外指定すると、その領域内のCB001が rows から消え、
    total_in_frame・ref_designator_countも減る。"""
    r = _region_results_fixture(excluded_names={'AWG20(BL)'})['a.dxf']
    labels_in_rows = {row['ラベル'] for row in r['rows']}
    assert labels_in_rows == {'U1'}
    assert r['total_in_frame'] == 1          # 2 -> 1（CB001除外）
    assert r['ref_designator_count'] == 1    # 2 -> 1


def test_region_results_excluded_region_rows_keep_pre_exclusion_count():
    """region_rows（領域一覧用）には除外領域の行が残り、mark='除外'、
    label_countは除外前の件数（1件）のまま。"""
    r = _region_results_fixture(excluded_names={'AWG20(BL)'})['a.dxf']
    by_name = {row['name']: row for row in r['region_rows']}
    assert by_name['AWG20(BL)']['mark'] == '除外'
    assert by_name['AWG20(BL)']['label_count'] == 1   # 除外前の件数
    assert by_name['RACK1']['mark'] == '特定'


def test_region_results_specified_region_unaffected_by_other_exclusion():
    """除外指定が他の領域に対するものなら、特定領域（RACK1）のラベルは
    影響を受けない。"""
    r = _region_results_fixture(excluded_names={'AWG20(BL)'})['a.dxf']
    by_label = {row['ラベル']: row for row in r['rows']}
    assert by_label['U1']['領域'] == 'RACK1'


# ---------------------------------------------------------------------------
# エンドツーエンド: 生成したExcelを openpyxl で読み戻す（表4）
# ---------------------------------------------------------------------------

def test_excel_output_reflects_exclusion_end_to_end():
    """除外ON時、各ファイルシート・抽出ラベル一覧からラベルが消え、
    領域一覧には行が残り判定列が'除外'、Summaryの総抽出ラベル数が減ることを
    実際に生成したExcelで確認する。"""
    region_results = _region_results_fixture(excluded_names={'AWG20(BL)'})
    xlsx = create_region_excel_output(region_results)
    wb = _load_wb(xlsx)

    file_ws = wb['a']
    file_labels = {row[1].value for row in file_ws.iter_rows(min_row=2)}
    assert 'CB001' not in file_labels
    assert 'U1' in file_labels

    total_ws = wb['抽出ラベル一覧']
    total_labels = {row[1].value for row in total_ws.iter_rows(min_row=2)}
    assert 'CB001' not in total_labels

    region_ws = wb['領域一覧']
    header = [c.value for c in region_ws[1]]
    assert header == ['ファイル名', '領域名', '判定', '面積率[%]',
                       '領域内ラベル数', '領域内機器符号候補数']
    rows_by_name = {row[1]: row for row in region_ws.iter_rows(min_row=2, values_only=True)}
    assert rows_by_name['AWG20(BL)'][2] == '除外'
    assert rows_by_name['AWG20(BL)'][4] == 1   # 除外前の件数

    summary_ws = wb['Summary']
    summary_header = [c.value for c in summary_ws[1]]
    total_col = summary_header.index('総抽出ラベル数')
    assert summary_ws.cell(row=2, column=total_col + 1).value == 1   # 2 -> 1


def test_excel_output_without_exclusion_matches_legacy_sheet_structure():
    """excluded_names 無指定（既定）でも領域一覧シートは6列（判定列が空欄）。"""
    region_results = _region_results_fixture()
    xlsx = create_region_excel_output(region_results)
    wb = _load_wb(xlsx)
    region_ws = wb['領域一覧']
    header = [c.value for c in region_ws[1]]
    assert header == ['ファイル名', '領域名', '判定', '面積率[%]',
                       '領域内ラベル数', '領域内機器符号候補数']
    rows_by_name = {row[1]: row for row in region_ws.iter_rows(min_row=2, values_only=True)}
    assert rows_by_name['AWG20(BL)'][2] in (None, '')
    assert rows_by_name['RACK1'][2] == '特定'
