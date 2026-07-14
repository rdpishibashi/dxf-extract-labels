"""端子一覧抽出（「端子一覧を抽出」オプション）の回帰テスト。

検証済み仕様:
  - 対象ファイル判定: タイトル（NFKC正規化後）に「UNIT内結線図」を含み、
    かつサブタイトルが「TB COMPONENT」でないこと
  - 候補ラベル判定: `TB`で始まりその直後に英大文字・数字が1文字以上続くもの
    （`TB取付板`のようにTBの直後が漢字のラベルは候補選定の時点で除外される）
  - 端子台矩形 = lineweight=25/color=2 のLINEで構成され、lineweight=50/color=4
    のCIRCLEによる橋渡しが1箇所以上ある矩形
  - 隣接する2矩形への上下衝突は「ラベルが矩形の上」を優先して解消する
    （EE6492-039-38A.dxf の TBP044/TBN241 衝突ケース）
  - 同一ラベルが複数の矩形にまたがる場合、端子番号は1つに集約される
    （EE6892-039-05B.dxf の TBLS ケース）
  - TB List（`build_terminal_rows`）は「端子台」でファイルをまたいでユニーク
    をとり、端子番号・図番を統合する（同じ番号が複数回登場する場合は
    `N(件数)` 表示、図番は複数ある場合カンマ区切りでABC順）

実 DXF（プロジェクト直下のサンプル、`sample-dxf/terminal-detector/`）を使った
ブラックボックステスト。サンプルが無い環境ではスキップする。
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from utils.terminal_detector import analyze_dxf_terminals, build_terminal_rows  # noqa: E402

SAMPLE_DIR = os.path.join(PROJECT_ROOT, 'sample-dxf')


def _find_sample(name):
    """指定ファイル名を sample-dxf/ 配下から再帰的に探す
    （`tests/regression/test_region_extraction.py` と同じ探索方式）。"""
    direct = os.path.join(SAMPLE_DIR, name)
    if os.path.exists(direct):
        return direct
    for dirpath, _dirnames, filenames in os.walk(SAMPLE_DIR):
        if name in filenames:
            return os.path.join(dirpath, name)
    return direct


SIMPLE = _find_sample('EE6892-039-02A.dxf')       # 単純ケース: TB001 -> [1, 2]
COLLISION = _find_sample('EE6492-039-38A.dxf')    # 隣接矩形の上下衝突を含む
MULTI_TERMINAL = _find_sample('EE6892-039-01B.dxf')  # 1矩形に10端子（複数箇所橋渡し）
AGGREGATE = _find_sample('EE6892-039-05B.dxf')    # 同名ラベルが複数矩形にまたがる
UNMATCHED = _find_sample('EE6492-039-90A.dxf')    # 対応する矩形が無いTBラベル
EXCLUDED = _find_sample('EE6892-039-82A.dxf')     # サブタイトル「TB COMPONENT」で対象外


def _entries_by_label(result):
    return {e['label']: e['numbers'] for e in result['entries']}


@pytest.mark.skipif(not os.path.exists(SIMPLE), reason="サンプルDXFが無い環境ではスキップ")
def test_simple_single_rectangle():
    result = analyze_dxf_terminals(SIMPLE, original_filename='EE6892-039-02A.dxf')
    assert result['is_target'] is True
    assert result['unmatched_labels'] == []
    assert _entries_by_label(result) == {'TB001 (27A)': [1, 2]}


@pytest.mark.skipif(not os.path.exists(COLLISION), reason="サンプルDXFが無い環境ではスキップ")
def test_adjacent_rectangle_collision_resolved():
    result = analyze_dxf_terminals(COLLISION, original_filename='EE6492-039-38A.dxf')
    assert result['is_target'] is True
    entries = _entries_by_label(result)
    # TBP044 と TBN241 は隣接する別々の矩形に正しく割り当てられ、
    # どちらの番号リストにも相手の端子番号が混入しない。
    assert 31 in entries['TBN241 (24A)']
    assert 31 not in entries['TBP044 (17.5A)']
    assert 4 in entries['TBP044 (17.5A)']
    assert 4 not in entries['TBN241 (24A)']


@pytest.mark.skipif(not os.path.exists(MULTI_TERMINAL), reason="サンプルDXFが無い環境ではスキップ")
def test_multi_terminal_rectangle():
    result = analyze_dxf_terminals(MULTI_TERMINAL, original_filename='EE6892-039-01B.dxf')
    assert result['is_target'] is True
    entries = _entries_by_label(result)
    assert entries['TB004 (50A)'] == list(range(1, 11))
    assert entries['TBM1 (50A)'] == [1, 2]


@pytest.mark.skipif(not os.path.exists(AGGREGATE), reason="サンプルDXFが無い環境ではスキップ")
def test_same_label_across_multiple_rectangles_is_aggregated():
    result = analyze_dxf_terminals(AGGREGATE, original_filename='EE6892-039-05B.dxf')
    assert result['is_target'] is True
    entries = _entries_by_label(result)
    # TBLS は2つの矩形([1,2]と[3])にまたがるが、1行に集約される
    assert entries['TBLS (17.5A)'] == [1, 2, 3]


@pytest.mark.skipif(not os.path.exists(UNMATCHED), reason="サンプルDXFが無い環境ではスキップ")
def test_kanji_suffix_label_excluded_at_candidate_stage():
    """「TB取付板」はTBの直後が漢字のため候補ラベル判定（`^TB[A-Z0-9]+`）に
    一致せず、候補選定の時点で除外される（unmatched_labelsにも現れない）。"""
    result = analyze_dxf_terminals(UNMATCHED, original_filename='EE6492-039-90A.dxf')
    assert result['is_target'] is True
    assert result['entries'] == []
    assert result['unmatched_labels'] == []


@pytest.mark.skipif(not os.path.exists(EXCLUDED), reason="サンプルDXFが無い環境ではスキップ")
def test_tb_component_subtitle_excludes_file():
    result = analyze_dxf_terminals(EXCLUDED, original_filename='EE6892-039-82A.dxf')
    assert result['is_target'] is False
    assert result['entries'] == []


@pytest.mark.skipif(
    not (os.path.exists(SIMPLE) and os.path.exists(COLLISION) and os.path.exists(EXCLUDED)),
    reason="サンプルDXFが無い環境ではスキップ")
def test_build_terminal_rows_excludes_tb_component_file_and_sorts_by_label():
    results = {
        'EE6892-039-02A.dxf': analyze_dxf_terminals(SIMPLE, original_filename='EE6892-039-02A.dxf'),
        'EE6492-039-38A.dxf': analyze_dxf_terminals(COLLISION, original_filename='EE6492-039-38A.dxf'),
        'EE6892-039-82A.dxf': analyze_dxf_terminals(EXCLUDED, original_filename='EE6892-039-82A.dxf'),
    }
    rows = build_terminal_rows(results)
    # 82A（TB COMPONENT）は対象外のため、その図番を持つ行は生成されない
    assert all(r['図番'] != 'EE6892-039-82A' for r in rows)
    # 「端子台」のABC順（TB001 < TBCMD < TBN241 < TBP044）
    assert [r['端子台'] for r in rows] == [
        'TB001 (27A)', 'TBCMD (17.5A)', 'TBN241 (24A)', 'TBP044 (17.5A)',
    ]


@pytest.mark.skipif(
    not (os.path.exists(COLLISION) and os.path.exists(AGGREGATE)),
    reason="サンプルDXFが無い環境ではスキップ")
def test_same_label_across_different_files_is_merged_in_tb_list():
    """TBN241 (24A) は EE6492-039-38A.dxf と EE6892-039-05B.dxf の両方に
    登場する（実データ）。TB List では1行に統合され、図番はABC順で
    カンマ区切り列挙される。"""
    results = {
        'EE6492-039-38A.dxf': analyze_dxf_terminals(COLLISION, original_filename='EE6492-039-38A.dxf'),
        'EE6892-039-05B.dxf': analyze_dxf_terminals(AGGREGATE, original_filename='EE6892-039-05B.dxf'),
    }
    rows = build_terminal_rows(results)
    tbn241_rows = [r for r in rows if r['端子台'] == 'TBN241 (24A)']
    assert len(tbn241_rows) == 1
    assert tbn241_rows[0]['図番'] == 'EE6492-039-38A, EE6892-039-05B'
    assert '12' in tbn241_rows[0]['端子No.']
    assert '48' in tbn241_rows[0]['端子No.']
