"""model/assembly_compare.py の単体テスト（モデル層のみ、Streamlit UI 非依存）。

組み合わせ表:
  - select_wiring_files: タイトルに「UNIT内結線図」を含む(半角)/含む(全角)/
    含まない/タイトルNone → A側として選ばれるかどうか
  - select_assembly_candidates: 0件/1件/複数件 → B側候補の件数
  - drawing_ident: 通常モードのキー(main_drawing_number)/領域モードのキー
    (drawing_number)/どちらも無し(ファイル名フォールバック)
  - compare_assembly: 複数A図面の合算・A図番数/A図番の結合、
    両方(個数一致)/両方(個数不一致)/A のみ/B のみ の区分、機器符号候補列、
    summaryの各カウント
  - row_style: 区分×個数一致/不一致の4ケース（表3）
  - _compare_key: 括弧なし/半角括弧/全角括弧/直前スペースあり(半角/全角)/
    括弧を含まない場合（2026-09-14 ユーザー指示）

実データ検証（2026-09-13、sample-dxf/組立図・内結線図_ME25-7103-0_ZMR1_339）で
A側のタイトルが全角「ＵＮＩＴ内結線図」（16件中13件）と半角「UNIT内結線図」
（3件）に混在することを確認済み。正規化を省くと大半を取りこぼすため、
select_wiring_files の全角タイトルケースは特に固定する（D13）。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from model import assembly_compare as ac


def _result(rows, main_drawing_number=None, drawing_number=None, title=None):
    data = {'rows': rows, 'title': title}
    if main_drawing_number is not None:
        data['main_drawing_number'] = main_drawing_number
    if drawing_number is not None:
        data['drawing_number'] = drawing_number
    return data


# ---------------------------------------------------------------------------
# select_wiring_files
# ---------------------------------------------------------------------------

def test_select_wiring_files_matches_halfwidth_title():
    results = {
        'a.dxf': _result([], title='UNIT内結線図'),
        'b.dxf': _result([], title='組立図'),
    }
    assert ac.select_wiring_files(results) == ['a.dxf']


def test_select_wiring_files_matches_fullwidth_title():
    """実データの主要ケース（16件中13件）: 全角『ＵＮＩＴ内結線図』も
    normalize_width() 経由でA側に選ばれること（D13）。"""
    results = {
        'a.dxf': _result([], title='ＵＮＩＴ内結線図'),
    }
    assert ac.select_wiring_files(results) == ['a.dxf']


def test_select_wiring_files_excludes_non_matching_title():
    results = {
        'a.dxf': _result([], title='ＳＹＳＴＥＭ＿Ｉ／Ｆ＿ＢＯＸ＿機器銘板'),
    }
    assert ac.select_wiring_files(results) == []


def test_select_wiring_files_excludes_none_title():
    results = {'a.dxf': _result([], title=None)}
    assert ac.select_wiring_files(results) == []


def test_select_wiring_files_sorted_and_multiple():
    results = {
        'z.dxf': _result([], title='UNIT内結線図'),
        'a.dxf': _result([], title='ＵＮＩＴ内結線図'),
        'm.dxf': _result([], title='組立図'),
    }
    assert ac.select_wiring_files(results) == ['a.dxf', 'z.dxf']


# ---------------------------------------------------------------------------
# select_assembly_candidates
# ---------------------------------------------------------------------------

def test_select_assembly_candidates_empty():
    results = {'a.dxf': _result([], title='UNIT内結線図')}
    assert ac.select_assembly_candidates(results) == []


def test_select_assembly_candidates_single():
    results = {
        'a.dxf': _result([], title='UNIT内結線図'),
        'b.dxf': _result([], title='ＳＹＳＴＥＭ＿組立図（ＢＫ）'),
    }
    assert ac.select_assembly_candidates(results) == ['b.dxf']


def test_select_assembly_candidates_multiple():
    results = {
        'b.dxf': _result([], title='組立図'),
        'c.dxf': _result([], title='組立図（別部品）'),
    }
    assert ac.select_assembly_candidates(results) == ['b.dxf', 'c.dxf']


# ---------------------------------------------------------------------------
# drawing_ident
# ---------------------------------------------------------------------------

def test_drawing_ident_uses_main_drawing_number_normal_mode():
    data = _result([], main_drawing_number='EE1234-500-01A')
    assert ac.drawing_ident('a.dxf', data) == 'EE1234-500-01A'


def test_drawing_ident_uses_drawing_number_region_mode():
    data = _result([], drawing_number='EE5678-500-01A')
    assert ac.drawing_ident('a.dxf', data) == 'EE5678-500-01A'


def test_drawing_ident_falls_back_to_filename_without_extension():
    data = _result([])
    assert ac.drawing_ident('a.dxf', data) == 'a'


# ---------------------------------------------------------------------------
# compare_assembly
# ---------------------------------------------------------------------------

def test_compare_assembly_aggregates_across_multiple_wiring_files():
    results = {
        'a1.dxf': _result(
            [{'機器符号候補': 'Y', 'ラベル': 'R10', '個数': 2}],
            main_drawing_number='EE001'),
        'a2.dxf': _result(
            [{'機器符号候補': 'Y', 'ラベル': 'R10', '個数': 3},
             {'機器符号候補': None, 'ラベル': 'タイトル', '個数': 1}],
            main_drawing_number='EE002'),
        'b.dxf': _result(
            [{'機器符号候補': 'Y', 'ラベル': 'R10', '個数': 5}],
            main_drawing_number='EE900'),
    }
    wiring_files = ['a1.dxf', 'a2.dxf']
    result = ac.compare_assembly(results, wiring_files, 'b.dxf')

    rows_by_label = {r['ラベル']: r for r in result['rows']}
    r10 = rows_by_label['R10']
    assert r10['A 合計出現数'] == 5
    assert r10['A 図番数'] == 2
    assert r10['A 図番'] == 'EE001, EE002'
    assert r10['B 出現数'] == 5
    assert r10['比較結果'] == ac.KUBUN_BOTH
    assert r10['機器符号候補'] == 'Y'

    title_row = rows_by_label['タイトル']
    assert title_row['比較結果'] == ac.KUBUN_A_ONLY
    assert title_row['B 出現数'] == 0
    assert title_row['機器符号候補'] is None

    assert result['b_file'] == 'b.dxf'
    assert result['summary'] == {
        'A 対象図番数': 2,
        'A ユニークラベル数': 2,
        'B ユニークラベル数': 1,
        '両方': 1,
        'A のみ': 1,
        'B のみ': 0,
    }


def test_compare_assembly_b_only_label():
    results = {
        'a.dxf': _result([{'機器符号候補': 'Y', 'ラベル': 'R10', '個数': 1}],
                          main_drawing_number='EE001'),
        'b.dxf': _result([{'機器符号候補': 'Y', 'ラベル': 'C1', '個数': 4}],
                          main_drawing_number='EE900'),
    }
    result = ac.compare_assembly(results, ['a.dxf'], 'b.dxf')
    rows_by_label = {r['ラベル']: r for r in result['rows']}
    assert rows_by_label['C1']['比較結果'] == ac.KUBUN_B_ONLY
    assert rows_by_label['C1']['A 合計出現数'] == 0
    assert rows_by_label['C1']['A 図番'] == ''
    assert result['summary']['B のみ'] == 1


def test_compare_assembly_no_wiring_files_yields_a_only_empty():
    results = {
        'b.dxf': _result([{'機器符号候補': 'Y', 'ラベル': 'R10', '個数': 1}],
                          main_drawing_number='EE900'),
    }
    result = ac.compare_assembly(results, [], 'b.dxf')
    assert result['summary']['A 対象図番数'] == 0
    assert result['summary']['A ユニークラベル数'] == 0
    assert result['rows'][0]['比較結果'] == ac.KUBUN_B_ONLY


# ---------------------------------------------------------------------------
# row_style（表3）
# ---------------------------------------------------------------------------

def test_row_style_a_only():
    assert ac.row_style(ac.KUBUN_A_ONLY, 3, 0) == ac.ROW_STYLE_A_ONLY


def test_row_style_b_only():
    assert ac.row_style(ac.KUBUN_B_ONLY, 0, 3) == ac.ROW_STYLE_B_ONLY


def test_row_style_both_matching_counts():
    assert ac.row_style(ac.KUBUN_BOTH, 5, 5) == ac.ROW_STYLE_MATCH


def test_row_style_both_mismatched_counts():
    assert ac.row_style(ac.KUBUN_BOTH, 5, 3) == ac.ROW_STYLE_MISMATCH


# ---------------------------------------------------------------------------
# _compare_key（2026-09-14 ユーザー指示: "(" 以降を無視して比較する）
# ---------------------------------------------------------------------------

def test_compare_key_no_parenthesis_returns_unchanged():
    assert ac._compare_key('R10') == 'R10'


def test_compare_key_halfwidth_parenthesis():
    assert ac._compare_key('R10(2.2K)') == 'R10'


def test_compare_key_fullwidth_parenthesis():
    assert ac._compare_key('FBWH（白）') == 'FBWH'


def test_compare_key_halfwidth_parenthesis_with_preceding_space():
    assert ac._compare_key('R10 (2.2K)') == 'R10'


def test_compare_key_fullwidth_parenthesis_with_preceding_fullwidth_space():
    assert ac._compare_key('R10　（2.2K）') == 'R10'


def test_compare_assembly_ignores_parenthesis_suffix_when_aggregating():
    """括弧内の補足（抵抗値等）だけが異なるラベルは、比較上は同一ラベルとして
    合算される（例: A側 'R10(2.2K)' とB側 'R10(5.1K)' は『両方』として一致）。
    比較シートの『ラベル』列も切り詰め後の 'R10' で表示される。"""
    results = {
        'a.dxf': _result([{'機器符号候補': 'Y', 'ラベル': 'R10(2.2K)', '個数': 2}],
                          main_drawing_number='EE001'),
        'b.dxf': _result([{'機器符号候補': 'Y', 'ラベル': 'R10(5.1K)', '個数': 2}],
                          main_drawing_number='EE900'),
    }
    result = ac.compare_assembly(results, ['a.dxf'], 'b.dxf')
    assert len(result['rows']) == 1
    row = result['rows'][0]
    assert row['ラベル'] == 'R10'
    assert row['比較結果'] == ac.KUBUN_BOTH
    assert row['A 合計出現数'] == 2
    assert row['B 出現数'] == 2
