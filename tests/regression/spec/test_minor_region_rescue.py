"""面積条件（既定5%）に満たない小領域を、他で採用された名称と同じなら採用する（2026-10-17 ユーザー指定）。

守る対象: 「領域一覧」「領域の確認」を経て全図面を通じて採用された領域名は、5%未満の小領域として
存在する図面でも領域として抽出する。例: EE4153-039-06B.dxf には 4.10% の `CN-IF2-1` があるが、
5%未満のため従来は no name のまま捨てられていた。

受入条件:
  - `analysis['minor_regions']` は `detect_minor_regions` が真のときだけ埋まる。`regions` の検出結果は、
    対辺補完で閉じた四角形のうち面積条件（`area_ratio`）を満たすものが加わる以外は不変（2026-10-27。
    加わった領域は通常領域と同じ扱いで、外側の領域の同名候補は外れる）
  - 採用対象は、「領域の確認」で選んだ名称も含む採用済みのすべての領域名（`adopted_region_names()`。アプリは常にこちら。
    2026-10-25 v3.17.2 でオプション「すべてのファイルに適用する」を廃止し、常時この動作にした）。
    `unconditional_region_names()`（「領域の確認」を経ずに決まる名称だけ）は、旧既定の振る舞いの関数としてテストが残っている。
    採用対象と一致する候補を持つ小領域だけを採用する
  - 同じ図面枠の同名の小領域は、すべて同じ名称で採用する
  - 候補が採用対象の2つ以上に一致すれば ` | ` で連結した名前で採用する。同枠の通常領域に採用済みの名称は名称単位で除く
  - 小領域の候補のどれかが「除外」指定されていれば追加しない（除外は通常領域にしか効かない）
  - 採用された小領域はラベルの `領域` 列・Excel「領域一覧」（判定=特定）・`regions_detected` に反映される
  - `minor_regions` を持たない解析結果（他プロジェクト・検出OFF）は何も変わらない
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
from model.region_detector import analyze_dxf_regions  # noqa: E402
from view.region_selection import build_region_cfg  # noqa: E402
import config  # noqa: E402

SAMPLE_DIR = os.path.join(PROJECT_ROOT, 'sample-dxf')
SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]


def _reg(rid, frame, *names, polygon=SQUARE):
    return {'id': rid, 'frame': frame, 'polygon': polygon, 'corners': polygon, 'area_pct': 50.0,
            'name_candidates': [(float(i), n) for i, n in enumerate(names)]}


def _analysis(regions, minors):
    return {'regions': regions, 'minor_regions': minors, 'frames': [(0, 100, 0, 100)]}


# ---------------------------------------------------------------------------
# add_minor_region_selections（採用対象 target_names は呼び出し側が決める）
# ---------------------------------------------------------------------------

def test_minor_region_adopted_when_name_is_a_target():
    analyses = {'a.dxf': _analysis([_reg(0, 0, 'SYSTEM I/F BOX')], [_reg(1, 0, 'CN-IF2-1')])}
    sel = {'a.dxf': {('a.dxf', 0): ['SYSTEM I/F BOX']}}
    out = ref_designator.add_minor_region_selections(analyses, sel, {'CN-IF2-1'})
    assert out['a.dxf'] == {('a.dxf', 0): ['SYSTEM I/F BOX'], ('a.dxf', 1): ['CN-IF2-1']}
    assert sel['a.dxf'] == {('a.dxf', 0): ['SYSTEM I/F BOX']}   # 引数は変更しない


def test_minor_region_not_added_when_no_name_is_a_target():
    analyses = {'a.dxf': _analysis([_reg(0, 0, 'SYSTEM I/F BOX')], [_reg(1, 0, 'TMGRES')])}
    sel = {'a.dxf': {('a.dxf', 0): ['SYSTEM I/F BOX']}}
    assert ref_designator.add_minor_region_selections(analyses, sel, {'CN-IF2-1'}) == sel
    assert ref_designator.add_minor_region_selections(analyses, sel, set()) == sel


def test_minor_region_not_added_when_same_frame_already_has_the_name():
    analyses = {'a.dxf': _analysis([_reg(0, 0, 'CN-IF2-1')], [_reg(1, 0, 'CN-IF2-1')])}
    sel = {'a.dxf': {('a.dxf', 0): ['CN-IF2-1']}}
    assert ref_designator.add_minor_region_selections(analyses, sel, {'CN-IF2-1'}) == sel


def test_minor_region_added_when_only_another_frame_has_the_name():
    analyses = {'a.dxf': _analysis([_reg(0, 0, 'CN-IF2-1')], [_reg(1, 1, 'CN-IF2-1')])}
    sel = {'a.dxf': {('a.dxf', 0): ['CN-IF2-1']}}
    out = ref_designator.add_minor_region_selections(analyses, sel, {'CN-IF2-1'})
    assert out['a.dxf'][('a.dxf', 1)] == ['CN-IF2-1']


def test_several_minor_regions_with_same_name_are_all_adopted():
    """同じ図面枠に同名の小領域が複数あれば、すべて同じ名称で採用する（2026-10-17 ユーザー決定）。"""
    analyses = {'a.dxf': _analysis([_reg(0, 0, 'BIG')],
                                   [_reg(1, 0, 'CN-IF2-1'), _reg(2, 0, 'CN-IF2-1')])}
    out = ref_designator.add_minor_region_selections(analyses, {'a.dxf': {('a.dxf', 0): ['BIG']}}, {'CN-IF2-1'})
    assert out['a.dxf'][('a.dxf', 1)] == ['CN-IF2-1'] and out['a.dxf'][('a.dxf', 2)] == ['CN-IF2-1']


def test_minor_region_matching_two_targets_is_adopted_with_or_delimiter():
    """候補が採用対象の2つ以上に一致する小領域は、候補順に ` | `（OR）で連結した名前で採用する。"""
    analyses = {'a.dxf': _analysis([_reg(0, 0, 'BIG')], [_reg(1, 0, 'CN-IF2-1', 'X', 'DCPS01')])}
    out = ref_designator.add_minor_region_selections(
        analyses, {'a.dxf': {('a.dxf', 0): ['BIG']}}, {'DCPS01', 'CN-IF2-1'})
    assert out['a.dxf'][('a.dxf', 1)] == ['CN-IF2-1 | DCPS01']
    assert ref_designator.OR_DELIMITER == ' | '


def test_name_already_in_same_frame_is_dropped_per_name_not_per_region():
    """同枠の通常領域に採用済みの名称だけを除き、残りの名称で採用する（名称単位）。"""
    analyses = {'a.dxf': _analysis([_reg(0, 0, 'CN-IF2-1')], [_reg(1, 0, 'CN-IF2-1', 'DCPS01')])}
    out = ref_designator.add_minor_region_selections(
        analyses, {'a.dxf': {('a.dxf', 0): ['CN-IF2-1']}}, {'CN-IF2-1', 'DCPS01'})
    assert out['a.dxf'][('a.dxf', 1)] == ['DCPS01']


def test_excluded_name_is_not_adopted_and_blocks_its_minor_regions():
    analyses = {'a.dxf': _analysis([_reg(0, 0, 'CN-IF2-1')], [_reg(1, 1, 'CN-IF2-1')])}
    sel = {'a.dxf': {('a.dxf', 0): ['CN-IF2-1']}}
    assert ref_designator.add_minor_region_selections(analyses, sel, {'CN-IF2-1'}, {'CN-IF2-1'}) == sel
    # 候補の一部だけが除外名称でも、その小領域は追加しない
    analyses = {'a.dxf': _analysis([_reg(0, 0, 'GOOD')], [_reg(1, 1, 'DROP', 'GOOD')])}
    sel = {'a.dxf': {('a.dxf', 0): ['GOOD']}}
    assert ref_designator.add_minor_region_selections(analyses, sel, {'GOOD', 'DROP'}, {'DROP'}) == sel


def test_width_normalization_and_adopted_region_names():
    analyses = {'a.dxf': _analysis([_reg(0, 0, 'X')], [_reg(1, 1, 'ＣＮ－ＩＦ２－１')])}
    sel = {'a.dxf': {('a.dxf', 0): ['']}}
    assert ref_designator.adopted_region_names(sel) == set()   # ['']（採用しない）は含まれない
    sel = {'a.dxf': {('a.dxf', 0): ['CN-IF2-1']}}
    assert ref_designator.adopted_region_names(sel) == {'CN-IF2-1'}
    out = ref_designator.add_minor_region_selections(analyses, sel, ref_designator.adopted_region_names(sel))
    assert out['a.dxf'][('a.dxf', 1)] == ['CN-IF2-1']


def test_analysis_without_minor_regions_is_unchanged():
    analyses = {'a.dxf': {'regions': [_reg(0, 0, 'A')], 'frames': []}}
    sel = {'a.dxf': {('a.dxf', 0): ['A']}}
    assert ref_designator.add_minor_region_selections(analyses, sel, {'A'}) == sel


# ---------------------------------------------------------------------------
# unconditional_region_names（「領域の確認」を経ずに決まる名称）
# ---------------------------------------------------------------------------

@pytest.fixture
def session():
    import streamlit as st
    original = st.session_state
    st.session_state = {}
    yield st.session_state
    st.session_state = original


def test_unconditional_names_exclude_manually_confirmed_ones(session):
    from view import region_selection as rs
    for n in ('A', 'B', 'C', 'D'):
        session[f'grc_{n}'] = True
    analyses = {'a.dxf': _analysis([
        _reg(0, 0, 'A'),                    # 候補1件 → 自動決定（含まれる）
        _reg(1, 0, 'B', 'C'),               # 2件チェック＝要選択。他図面に決定名なし → 含まれない
        _reg(2, 1, 'D', 'C'),               # 要選択。別の図面枠（frame 0）の決定名 A とは無関係 → 含まれない
    ], [])}
    assert rs.unconditional_region_names(analyses) == {'A'}


def test_unconditional_names_include_other_frame_preference(session):
    from view import region_selection as rs
    for n in ('SYSTEM I/F BOX', 'CN-IF2-1'):
        session[f'grc_{n}'] = True
    analyses = {'a.dxf': _analysis([
        _reg(0, 0, 'SYSTEM I/F BOX', 'CN-IF2-1'),   # 要選択だが他図面で SYSTEM I/F BOX が決定
        _reg(1, 1, 'SYSTEM I/F BOX'),
    ], [])}
    assert rs.unconditional_region_names(analyses) == {'SYSTEM I/F BOX'}


def test_unconditional_names_skip_excluded_regions(session):
    from view import region_selection as rs
    session['grc_A'] = True
    session['gre_A'] = True
    analyses = {'a.dxf': _analysis([_reg(0, 0, 'A')], [])}
    assert rs.unconditional_region_names(analyses) == set()


# ---------------------------------------------------------------------------
# 出力（named / 領域一覧 / regions_detected / Excel）への反映
# ---------------------------------------------------------------------------

def test_adopted_minor_region_reaches_labels_and_excel():
    big = [(0, 0), (100, 0), (100, 100), (0, 100)]
    small = [(10, 10), (20, 10), (20, 20), (10, 20)]
    analysis = _analysis([_reg(0, 0, 'BIG', polygon=big)], [_reg(5, 0, 'CN-IF2-1', polygon=small)])
    analysis['regions'][0]['area_pct'] = 90.0
    analysis['minor_regions'][0]['area_pct'] = 4.0
    analyses = {'a.dxf': analysis}
    data = {'a.dxf': {'labels': [('R1X', 15, 15), ('R2X', 50, 50)], 'total_in_frame': 2}}
    sel = {'a.dxf': {('a.dxf', 0): ['BIG']}}
    other = {'b.dxf': _analysis([_reg(0, 0, 'CN-IF2-1')], [])}
    sel_all = ref_designator.add_minor_region_selections(
        {**analyses, **other}, {**sel, 'b.dxf': {('b.dxf', 0): ['CN-IF2-1']}}, {'CN-IF2-1'})
    results = ref_designator.build_ref_designator_region_results(data, analyses, {'a.dxf': sel_all['a.dxf']})
    r = results['a.dxf']
    assert r['regions_detected'] == 2
    by_label = {row['ラベル']: row['領域'] for row in r['rows']}
    assert by_label == {'R1X': 'BIG, CN-IF2-1', 'R2X': 'BIG'}
    rows = {x['name']: x for x in r['region_rows']}
    assert set(rows) == {'BIG', 'CN-IF2-1'} and rows['CN-IF2-1']['mark'] == '特定'
    wb = openpyxl.load_workbook(BytesIO(create_region_excel_output(results)))
    names = [row[1] for row in wb['領域一覧'].iter_rows(min_row=2, values_only=True)]
    assert sorted(names) == ['BIG', 'CN-IF2-1']


def test_unadopted_minor_region_is_invisible_in_output():
    analysis = _analysis([_reg(0, 0, 'BIG')], [_reg(5, 0, 'TMGRES')])
    data = {'a.dxf': {'labels': [('R1X', 5, 5)], 'total_in_frame': 1}}
    results = ref_designator.build_ref_designator_region_results(
        data, {'a.dxf': analysis}, {'a.dxf': {('a.dxf', 0): ['BIG']}})
    assert results['a.dxf']['regions_detected'] == 1
    assert [x['name'] for x in results['a.dxf']['region_rows']] == ['BIG']


# ---------------------------------------------------------------------------
# 実データ（EE6892-039-03A: 図面1に 2.41% の CN-IF2-1）
# ---------------------------------------------------------------------------

def _find_sample(name):
    for root, _dirs, files in os.walk(SAMPLE_DIR):
        if name in files:
            return os.path.join(root, name)
    return None


@pytest.mark.slow
def test_real_data_minor_regions_do_not_change_regions():
    path = _find_sample('EE6892-039-03A.dxf')
    if path is None:
        pytest.skip('サンプル DXF が無い')
    cfg = build_region_cfg(config.region_detection_config)
    plain = analyze_dxf_regions(path, cfg)
    with_minor = analyze_dxf_regions(path, {**cfg, 'detect_minor_regions': True})
    assert plain['minor_regions'] == []
    key = lambda a: [(r['id'], r['frame'], r['polygon'], r['name_candidates']) for r in a['regions']]  # noqa: E731
    assert key(plain) == key(with_minor)
    minors = {(r['frame'], r['name_candidates'][0][1]) for r in with_minor['minor_regions']}
    assert (0, 'CN-IF2-1') in minors
    ids = [r['id'] for r in with_minor['regions']] + [r['id'] for r in with_minor['minor_regions']]
    assert len(ids) == len(set(ids))


@pytest.mark.slow
def test_real_data_rectangle_completion_promotes_large_region_to_regions():
    """06B（`SENSOR I/F B.D-Ⅲ` 枠の左辺が部品箱で途切れ、上側の片が長さ2未満）でも、対辺補完で閉じた
    領域が見つかる。小さい方（`SENSOR I/F B.D-Ⅲ` 0.9%）は `minor_regions` に入る。面積条件を満たす方
    （`CN-IF2-1` 12.7%）は、他で採用された名称との一致を待たず通常領域（`regions`）になる
    （2026-10-27 ユーザー決定。それまでは `minor_regions` に入り、`CN-IF2-1` が他図面で採用されない限り
    出力に載らなかった）。外側の領域0の候補からは同名の `CN-IF2-1` が外れる。
    補完を使わない検出（`detect_minor_regions` なし）の `regions` は変わらない。"""
    path = _find_sample('EE3273-039-06B.dxf')
    if path is None:
        pytest.skip('サンプル DXF が無い')
    cfg = build_region_cfg(config.region_detection_config)
    plain = analyze_dxf_regions(path, cfg)
    with_minor = analyze_dxf_regions(path, {**cfg, 'detect_minor_regions': True})
    in_range = lambda a: [r for r in a['regions'] if 12 < r['area_pct'] < 13]  # noqa: E731
    assert in_range(plain) == []
    promoted = in_range(with_minor)
    assert [(r['frame'], [t for _d, t in r['name_candidates']]) for r in promoted] == [(0, ['CN-IF2-1'])]
    outer = [r for r in with_minor['regions'] if r['area_pct'] > 70][0]
    assert [t for _d, t in outer['name_candidates']] == ['SYSTEM I/F BOX']
    minors = {(r['frame'], r['name_candidates'][0][1]): r['area_pct'] for r in with_minor['minor_regions']}
    assert minors.keys() == {(0, 'SENSOR I/F B.D-Ⅲ')} and minors[(0, 'SENSOR I/F B.D-Ⅲ')] < 5
    ids = [r['id'] for r in with_minor['regions']] + [r['id'] for r in with_minor['minor_regions']]
    assert len(ids) == len(set(ids))
