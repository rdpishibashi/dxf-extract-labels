"""clean_mtext_format_codes（ezdxf plain_mtext ベース）の回帰テスト。

v1.5.1（2026-06-15）で MTEXT 整形を手書き正規表現から
`ezdxf.tools.text.plain_mtext` へ移行した。実データ 12,145 件
（EE6868-500-01C.dxf / EE6888-602-01A.dxf）で旧実装と出力一致を確認済み。

本テストは:
  1. 旧実装と同じ挙動を維持すべきケース（後方互換）
  2. 新実装で改善されたケース（旧実装では未対応だったコード）
の期待値を固定し、将来の退行を防ぐ。
"""
from utils.extract_labels import clean_mtext_format_codes as c


# --- 後方互換: 旧実装と同じ出力を維持すべきケース --------------------------

def test_empty_or_none():
    assert c("") == ""
    assert c(None) == ""


def test_plain_text_passthrough():
    assert c("ABC123") == "ABC123"


def test_font_code_stripped():
    assert c(r"\fArial|b1|i0;HEATER") == "HEATER"


def test_height_and_color_stripped():
    assert c(r"\H100;\C1;ABC") == "ABC"


def test_alignment_stripped():
    assert c(r"\A1;ABC") == "ABC"


def test_paragraph_break_to_space():
    assert c(r"LINE1\PLINE2") == "LINE1 LINE2"


def test_yen_sign_normalized():
    # 日本語環境の円マークはバックスラッシュとして解釈される
    assert c("¥H100;ABC") == "ABC"


def test_whitespace_collapsed():
    assert c("A   B") == "A B"


# --- 改善: 旧実装では未対応だったコード（新実装で正しく処理）---------------

def test_special_char_diameter():
    assert c("%%c100") == "Ø100"


def test_special_char_degree():
    assert c("45%%d") == "45°"


def test_special_char_plus_minus():
    assert c("12%%p3") == "12±3"


def test_fraction_preserved():
    # 旧実装は分数を脱落させ "1" になっていた
    assert c(r"\A1;1\S1/2;") == "11/2"


def test_caret_sequence_to_space():
    # 旧実装は ^I を素通しで残していた
    assert c("TEXT^I TAB") == "TEXT TAB"


def test_braces_are_grouping_not_content():
    # 中括弧はフォーマットグループの区切りで内容ではない
    assert c(r"{\C1;RED}TEXT") == "REDTEXT"
