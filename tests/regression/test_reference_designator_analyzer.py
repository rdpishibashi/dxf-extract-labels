"""tools/reference_designator_analyzer.py の回帰テスト。

パターン・除外リストの定義自体は utils/ref_designator.py（DXF-extract-labels
本体アプリの判定ロジック）を単一の正として参照するため、ここでは集計・
パターン表記生成・Excel 出力の配線を検証する（定義の重複検証はしない）。
"""
import os
import sys
from io import BytesIO

import openpyxl
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from tools import reference_designator_analyzer as rda  # noqa: E402


# ---------------------------------------------------------------------------
# build_pattern_signature
# ---------------------------------------------------------------------------

def test_pattern_signature_letters_only():
    assert rda.build_pattern_signature('ACTAPEFEM', 'letters_only') == 'ACTAPEFEM'


def test_pattern_signature_no_tail_collapses_to_generic_digits():
    assert rda.build_pattern_signature('ACTAPEFEM2', 'letters_digits_any') == 'ACTAPEFEM+1*'


def test_pattern_signature_with_tail_keeps_digit_count():
    assert rda.build_pattern_signature('AAC1B4-07', 'hyphen_letters_digits_any') is None
    # AAC1B4-07 はハイフンパターンではなく letters_digits_any（先頭がAAC+1桁の数字）
    assert rda.build_pattern_signature('AAC1B4-07', 'letters_digits_any') == 'AAC1+A*1*-1*'


def test_pattern_signature_hyphen_pattern():
    assert rda.build_pattern_signature('CN-IF2-1', 'hyphen_letters_digits_any') == 'CN-IF1+-1*'


# ---------------------------------------------------------------------------
# aggregate_labels / classify_aggregated_labels（合成 xlsx）
# ---------------------------------------------------------------------------

def _make_total_xlsx(rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Total'
    ws.append(['ラベル', '個数'])
    for label, count in rows:
        ws.append([label, count])
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_aggregate_labels_merges_zenkaku_hankaku_and_sums_across_files():
    src_a = _make_total_xlsx([('R10', 3), ('ＲＡＣＫ１', 2)])
    src_b = _make_total_xlsx([('R10', 4), ('GND', 1)])
    agg, stats = rda.aggregate_labels([('a.xlsx', src_a), ('b.xlsx', src_b)])
    assert agg['R10']['count'] == 7
    assert agg['R10']['files'] == {'a.xlsx', 'b.xlsx'}
    assert agg['RACK1']['count'] == 2   # 全角→半角正規化
    assert stats == [
        {'ファイル名': 'a.xlsx', 'ラベル種類数': 2},
        {'ファイル名': 'b.xlsx', 'ラベル種類数': 2},
    ]


def test_classify_aggregated_labels_splits_candidate_excluded_and_drops_no_match():
    src = _make_total_xlsx([
        ('R10', 5), ('CN3', 2), ('GND', 9), ('(2/5)', 1), ('AWG14', 3),
    ])
    agg, _ = rda.aggregate_labels([('x.xlsx', src)])
    ref_rows, sig_rows, impact = rda.classify_aggregated_labels(agg)

    by_label = {r['ラベル']: r for r in ref_rows}
    assert '(2/5)' not in by_label   # 3パターンいずれにも不一致 -> 完全に除外
    assert by_label['R10']['除外ステータス'] == ''
    assert by_label['GND']['除外ステータス'] == '確定'
    assert by_label['GND']['除外カテゴリ'] == 'circuit_description'
    assert by_label['AWG14']['除外カテゴリ'] == 'wire_gauge'

    assert impact['circuit_description'] == {'labels': 1, 'count': 9}
    assert impact['wire_gauge'] == {'labels': 1, 'count': 3}


# ---------------------------------------------------------------------------
# build_output_workbook
# ---------------------------------------------------------------------------

def test_build_output_workbook_sheet_names_and_remaining_sorted_desc():
    src = _make_total_xlsx([('R10', 2), ('CN3', 9), ('GND', 5)])
    agg, _ = rda.aggregate_labels([('x.xlsx', src)])
    ref_rows, sig_rows, impact = rda.classify_aggregated_labels(agg)
    data = rda.build_output_workbook(ref_rows, sig_rows, impact)

    wb = openpyxl.load_workbook(BytesIO(data))
    assert wb.sheetnames == [
        'ReferenceDesignators', 'Patterns', 'PatternSignatures',
        'ExclusionPatterns', 'RemainingUnclassified',
    ]

    remaining = list(wb['RemainingUnclassified'].iter_rows(min_row=2, values_only=True))
    assert [r[0] for r in remaining] == ['CN3', 'R10']   # 個数降順、GNDは含まれない

    ref_labels = {r[0] for r in wb['ReferenceDesignators'].iter_rows(min_row=2, values_only=True)}
    assert ref_labels == {'R10', 'CN3', 'GND'}


def test_build_output_workbook_empty_input_does_not_raise():
    data = rda.build_output_workbook([], [], {})
    wb = openpyxl.load_workbook(BytesIO(data))
    assert 'ReferenceDesignators' in wb.sheetnames


# ---------------------------------------------------------------------------
# _iter_input_sources（フォルダ探索）
# ---------------------------------------------------------------------------

def test_iter_input_sources_combines_uploads_and_folder(tmp_path):
    (tmp_path / 'extracted_labels.xlsx').write_bytes(b'dummy')
    (tmp_path / 'extracted_labels (1).xlsx').write_bytes(b'dummy')
    (tmp_path / '~$extracted_labels.xlsx').write_bytes(b'dummy')   # ロックファイル除外
    (tmp_path / 'other.xlsx').write_bytes(b'dummy')                # パターン非一致除外

    class _FakeUpload:
        def __init__(self, name, content):
            self.name = name
            self._content = content

        def getvalue(self):
            return self._content

    uploads = [_FakeUpload('uploaded.xlsx', b'uploaded-bytes')]
    sources = rda._iter_input_sources(uploads, [str(tmp_path)], 'extracted_labels*.xlsx', False)
    names = sorted(n for n, _ in sources)
    assert names == ['extracted_labels (1).xlsx', 'extracted_labels.xlsx', 'uploaded.xlsx']


def test_iter_input_sources_missing_folder_is_skipped_not_raised():
    sources = rda._iter_input_sources([], ['/no/such/folder'], 'extracted_labels*.xlsx', False)
    assert sources == []
