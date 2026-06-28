"""filter_non_circuit_symbols / validate_circuit_symbols の回帰テスト。

common_utils.py の機器符号フィルタリングロジックを直接検証する。
パターン定義（正規表現）の変更が既存の判定結果を変えた場合にすぐ気づけるよう、
実データで確認済みのケースを固定している。
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from utils.common_utils import filter_non_circuit_symbols, validate_circuit_symbols  # noqa: E402


# ---------------------------------------------------------------------------
# filter_non_circuit_symbols — 機器符号に一致するラベルを通す
# ---------------------------------------------------------------------------

def test_empty_input():
    matched, excluded = filter_non_circuit_symbols([])
    assert matched == []
    assert excluded == 0


def test_basic_letter_only():
    matched, excluded = filter_non_circuit_symbols(['FB', 'CNCNT', 'MSS'])
    assert matched == ['FB', 'CNCNT', 'MSS']
    assert excluded == 0


def test_letter_plus_number():
    matched, excluded = filter_non_circuit_symbols(['R10', 'CN3', 'PSW1'])
    assert matched == ['R10', 'CN3', 'PSW1']
    assert excluded == 0


def test_letter_number_letter():
    matched, excluded = filter_non_circuit_symbols(['X14A', 'RMSS2A', 'U23B'])
    assert matched == ['X14A', 'RMSS2A', 'U23B']
    assert excluded == 0


def test_bracketed_forms():
    matched, excluded = filter_non_circuit_symbols(['FB()', 'MSS(MOTOR)', 'R10(2.2K)', 'U23B(DAC)'])
    assert matched == ['FB()', 'MSS(MOTOR)', 'R10(2.2K)', 'U23B(DAC)']
    assert excluded == 0


def test_space_containing_labels_are_excluded():
    """スペースを含む文字列はどのパターンにも一致しない → 除外される"""
    labels = ['MPD RACK1', 'HEATER CTRL', 'A CHAMBER']
    matched, excluded = filter_non_circuit_symbols(labels)
    assert matched == []
    assert excluded == 3


def test_pure_letter_strings_pass_through():
    """英字のみ2文字以上は ^[A-Za-z]{2,}$ に一致 → 機器符号として通過する
    （CHAMBER や NOTE が通るのはこの関数の仕様。
      領域名としての除外は上位の _is_valid_name_candidate が担う）"""
    matched, excluded = filter_non_circuit_symbols(['CHAMBER', 'NOTE', 'FB'])
    assert set(matched) == {'CHAMBER', 'NOTE', 'FB'}
    assert excluded == 0


def test_mixed_input_counts():
    labels = ['R10', 'CHAMBER BAKE', 'CN3', 'NOTE', 'FB']
    matched, excluded = filter_non_circuit_symbols(labels)
    # 'CHAMBER BAKE'（スペースあり）だけが除外される
    assert set(matched) == {'R10', 'CN3', 'NOTE', 'FB'}
    assert excluded == 1


def test_single_letter_excluded():
    """英字1文字は 2文字以上パターンに非一致 → 除外"""
    matched, excluded = filter_non_circuit_symbols(['A', 'B', 'R'])
    assert matched == []
    assert excluded == 3


def test_rack_label_excluded():
    """RACK1 は機器符号パターン（英字+数字）に一致 → 通過する
    （呼び出し側が circuit_keep_terms で保護するが、この関数自体はマッチさせる）"""
    matched, excluded = filter_non_circuit_symbols(['RACK1'])
    assert matched == ['RACK1']
    assert excluded == 0


def test_numbers_only_excluded():
    matched, excluded = filter_non_circuit_symbols(['123', '001'])
    assert matched == []
    assert excluded == 2


def test_return_order_preserved():
    """入力順が維持される"""
    labels = ['CN3', 'R10', 'FB']
    matched, excluded = filter_non_circuit_symbols(labels)
    assert matched == ['CN3', 'R10', 'FB']


# ---------------------------------------------------------------------------
# validate_circuit_symbols — 標準パターン非準拠の機器符号を返す
# ---------------------------------------------------------------------------

def test_validate_all_valid():
    """標準パターンに準拠したラベルは空リストを返す"""
    labels = ['CB001', 'R10', 'C1', 'L5', 'Q2', 'U1A', 'SW1', 'CN3']
    invalid = validate_circuit_symbols(labels)
    assert invalid == []


def test_validate_detects_invalid():
    """標準パターン外のラベルが報告される"""
    labels = ['CB001', 'UNKNOWN_SYMBOL', 'R10', 'WEIRD123XYZ']
    invalid = validate_circuit_symbols(labels)
    assert 'UNKNOWN_SYMBOL' in invalid
    assert 'WEIRD123XYZ' in invalid
    assert 'CB001' not in invalid
    assert 'R10' not in invalid


def test_validate_empty():
    assert validate_circuit_symbols([]) == []


def test_validate_circuit_breaker_variants():
    labels = ['CB001', 'MCCB001', 'NFB001', 'ELB(CB)001']
    invalid = validate_circuit_symbols(labels)
    assert invalid == []
