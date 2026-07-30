"""機器符号パターン分類ロジック（model/ref_designator_patterns.py）の回帰テスト。

**2026-07-30 移設の経緯**: `model/ref_designator.py`（DXF抽出パイプライン）は
判定条件の簡素化により、本モジュールの除外・追加・確定パターンを使わなく
なった（3つの候補パターンに一致すれば全件そのまま出力する方式に変更。
`test_ref_designator.py` 参照）。しかし本モジュール自体は
`region_detector.py` の領域名候補フィルタが引き続き利用しているため、
このファイルの回帰テストは削除せず、対象モジュールを `ref_designator`
から `ref_designator_patterns` に付け替えて維持する。

検証済み仕様（2026-07-26 確定、reference_designator_candidates.xlsx が正）:
  - パターン: 英字繰返し(-英字繰返し)?+数字繰返し+英数字/ハイフン任意、または英字繰返しのみ
  - 判定は括弧より前の部分に対して行う（R10(2.2K) → R10 で判定）
  - 除外: 単一英大文字・末尾+-・AWG・RACK*・図番・JIS/DWG*・A/B+数字・PE+数字・
    L/N/P+数字・X+英字・数字4桁以上連続（配線ラベル）、および
    普通名詞/回路説明/ユニット名/ケーブル色/図面情報枠語の完全一致リスト
  - 追加パターン（reference_deginator_pattern_added.txt、2026-07-26）
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model import ref_designator_patterns as rdp  # noqa: E402


# ---------------------------------------------------------------------------
# is_ref_designator_candidate — パターン一致
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('label', [
    'R10', 'CN3', 'D4', 'CN1',                   # 英字+数字（A/Bは機器端子として除外対象）
    'FB', 'CNCNT', 'MSS',                        # 英字のみ
    'CN-IF2-1', 'AAC1B4-07', 'AB-CD12',           # 英字-英字+数字(+ハイフン任意)
    'R10(2.2K)', 'FB()', 'MSS(MOTOR)', 'U23B(DAC)',  # 括弧付き（括弧前で判定）
])
def test_candidate_pattern_matches(label):
    assert rdp.is_ref_designator_candidate(label) is True


@pytest.mark.parametrize('label', [
    '123',        # 数字のみ（英字繰返しが先頭に必須）
    '',           # 空文字列
    '(BU)',       # 括弧が先頭（判定対象が空になる）
    'R 10',       # スペースを含む
    'あいう',      # 英数字以外
])
def test_candidate_pattern_rejects(label):
    assert rdp.is_ref_designator_candidate(label) is False


# ---------------------------------------------------------------------------
# is_ref_designator_candidate — 除外パターン（ExclusionPatterns シートが正）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('label,reason', [
    ('A', 'single_letter_position'),
    ('R10-', 'trailing_sign'),
    ('AWG14', 'wire_gauge'),
    ('AWG', 'wire_gauge'),
    ('RACK1', 'rack_prefix'),
    ('RACK1-2', 'rack_prefix'),
    ('EE1234-500-01A', 'drawing_number'),
    ('DE3527-553-05B', 'drawing_number'),
    ('JIS123', 'jis_dwg_prefix'),
    ('DWGNO', 'jis_dwg_prefix'),
    ('A1', 'terminal_row_letter_digit'),
    ('B12', 'terminal_row_letter_digit'),
    ('PE1', 'earth_terminal_digit'),
    ('L1', 'phase_rail_letter_digit'),
    ('N24', 'phase_rail_letter_digit'),
    ('P24', 'phase_rail_letter_digit'),
    ('XRST', 'io_signal_x_prefix'),
    ('XMCON', 'io_signal_x_prefix'),
    ('GND', 'circuit_description'),
    ('POWER', 'circuit_description'),
    ('ALARM', 'common_nouns'),
    ('MOTOR', 'common_nouns'),
    ('CTC', 'unit_names'),
    ('SHIELD', 'unit_names'),
    ('BK', 'cable_colors'),
    ('RD', 'cable_colors'),
    ('TITLE', 'titleblock_terms'),
    ('DATE', 'titleblock_terms'),
    ('W1234', 'wiring_digit_run'),
    ('ETC-JP2', 'added_excl:ETC-JPn*'),
])
def test_exclusion_categories(label, reason):
    assert rdp.is_ref_designator_candidate(label) is False, (
        f'{label} should be excluded ({reason})')


def test_x_plus_digit_is_not_excluded():
    """X+数字（X1, X24）はIEC上の正規の端子/コネクタ記号であり、
    X+英字（io_signal_x_prefix）とは区別して候補に残す。"""
    assert rdp.is_ref_designator_candidate('X1') is True
    assert rdp.is_ref_designator_candidate('X24') is True


@pytest.mark.parametrize('label', ['OUT2', 'IN1', 'COM3', 'GND9', 'POWER5'])
def test_circuit_description_excludes_keyword_plus_single_digit(label):
    """circuit_description のキーワードは末尾に数字1桁があっても除外対象
    （例 OUT2, IN1, COM3。2026-07-10 ユーザー指摘）。"""
    assert rdp.is_ref_designator_candidate(label) is False
    assert rdp.classify_judgment_detailed(label) == ('excluded', 'circuit_description')


@pytest.mark.parametrize('label', ['GND12'])
def test_circuit_description_two_digit_suffix_not_excluded(label):
    """数字2桁以上は circuit_description の除外対象外（候補として残る）。

    OUT12/IN10 はかつてこの対象だったが、追加除外パターン OUTn*/I.*
    （reference_deginator_pattern_added.txt、2026-07-26）が桁数を問わず
    除外するため対象から外れた。GND12 は追加除外リストに GND 系の
    プレフィックスが無いため、引き続き候補として残る
    （`test_added_exclusion_overrides_two_digit_suffix_exception` 参照）。
    """
    status, _category = rdp.classify_judgment_detailed(label)
    assert status != 'excluded'


@pytest.mark.parametrize('label,category', [
    ('OUT12', 'added_excl:O.*'),   # O.* が OUTn* より先に定義されており先に一致
    ('IN10', 'added_excl:I.*'),
])
def test_added_exclusion_overrides_two_digit_suffix_exception(label, category):
    """追加除外パターン（reference_deginator_pattern_added.txt、2026-07-26）は
    桁数を問わず除外するため、OUT12/IN10 は 2026-07-10 の「数字2桁以上は
    circuit_description の除外対象外」という既存方針の例外だったが、今回
    追加リストを優先する方針（2026-07-26 ユーザー確定）でこの2件は除外に
    転じた。"""
    status, actual_category = rdp.classify_judgment_detailed(label)
    assert status == 'excluded'
    assert actual_category == category


# ---------------------------------------------------------------------------
# 除外パターン追加分（2026-07-10: circuit_description 追加語 / phase_rail 拡張）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('label', [
    'AOUT', 'LG', 'MR', 'MRR', 'RX', 'TX', 'CLR', 'ZERO', 'YOUT', 'AG',
])
def test_circuit_description_new_keywords_excluded(label):
    assert rdp.classify_judgment_detailed(label) == ('excluded', 'circuit_description')


@pytest.mark.parametrize('label', ['AG1', 'ZERO1', 'RX9'])
def test_circuit_description_new_keywords_plus_single_digit_excluded(label):
    assert rdp.classify_judgment_detailed(label) == ('excluded', 'circuit_description')


@pytest.mark.parametrize('label', ['ZERO12', 'AG12'])
def test_circuit_description_new_keywords_two_digit_suffix_not_excluded(label):
    status, _category = rdp.classify_judgment_detailed(label)
    assert status != 'excluded'


@pytest.mark.parametrize('label', ['N24AB', 'L1XY', 'P24Z', 'N1A'])
def test_phase_rail_letter_digit_allows_repeated_trailing_letters(label):
    """L/N/P+数字繰り返し+英大文字繰り返し（2026-07-10: 末尾の英大文字を
    0字以上に拡張。従来は0-1字までしか許容していなかった）。"""
    assert rdp.classify_judgment_detailed(label) == (
        'excluded', 'phase_rail_letter_digit')


def test_person_name_not_excluded_by_content_filter():
    """人名リストは実装に持たない（`ref_designator.py` 側の図面情報枠の
    幾何検出で対応する設計）ため、単体の is_ref_designator_candidate は
    人名を弾かない。"""
    assert rdp.is_ref_designator_candidate('KURIHARA') is True


# ---------------------------------------------------------------------------
# 除外パターン追加分（2026-07-11: wiring_digit_run＝数字4桁以上連続の配線ラベル）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('label', [
    'W1234', 'CN2345', 'D1000', 'R1000', 'AB1234-5', 'GND1234',
])
def test_wiring_digit_run_excludes_four_or_more_consecutive_digits(label):
    assert rdp.classify_judgment_detailed(label) == (
        'excluded', 'wiring_digit_run')


@pytest.mark.parametrize('label', ['R100', 'D100', 'CN123', 'X999'])
def test_wiring_digit_run_allows_up_to_three_consecutive_digits(label):
    """数字3桁までは配線ラベル除外の対象外（候補として残る）。"""
    status, _category = rdp.classify_judgment_detailed(label)
    assert status != 'excluded'


def test_wiring_digit_run_requires_digits_to_be_consecutive():
    """ハイフン等で分断された数字は合算しない（連続4桁以上のみ対象）。
    'AB12-34' は '12'・'34' がそれぞれ2桁ずつで、連続した4桁は存在しない。"""
    status, _category = rdp.classify_judgment_detailed('AB12-34')
    assert status != 'excluded'


def test_wiring_digit_run_ignores_parenthetical_digits():
    """括弧内の数字は判定対象外（括弧より前の判定用文字列のみで判定）。
    'R10(1234)' は判定用文字列が 'R10' のため除外されない。"""
    assert rdp.is_ref_designator_candidate('R10(1234)') is True


# ---------------------------------------------------------------------------
# 確定パターン（CONFIRMED_PATTERN_CATEGORIES / matched_confirmed_category /
# is_confirmed_designator）— region_detector.py の領域名候補フィルタが利用
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('label,expected', [
    ('R10', 'letters_digits_2or3'),
    ('R100', 'letters_digits_2or3'),
    ('CN3', 'cn_single_digit'),           # CN+数字1桁（2026-07-10 追加）
    ('R10A', 'letters_digits_2or3_letter'),
    ('CN-IF21', 'cn_if_prefix'),          # "CN-IF"+任意の文字が優先判定される
    ('CN-IF2-1', 'cn_if_prefix'),         # 2026-07-10: 末尾に続きがあっても確定対象
    ('D1', 'single_letter_digits_except_ab'),
    ('D1000', 'single_letter_digits_except_ab'),   # 桁数不問
    ('A1', None),                         # A,Bは対象外（単一英字パターンのみ）
    ('B12', 'letters_digits_2or3'),       # A,B除外は単一英字パターン限定
    ('FB', None),                         # 英字のみ（数字を要求するパターンはいずれも不一致）
])
def test_matched_confirmed_category(label, expected):
    assert rdp.matched_confirmed_category(label) == expected


# ---------------------------------------------------------------------------
# 確定パターン追加分（2026-07-10: CN+数字1桁 / "CN-IF"+任意 / R(...) / VR(...)）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('label,expected', [
    ('CN1', 'cn_single_digit'),
    ('CN9', 'cn_single_digit'),
    ('CN12', 'letters_digits_2or3'),          # 数字2桁は cn_single_digit の対象外
    ('CN-IF1-1A', 'cn_if_prefix'),
    ('CN-IFX', 'cn_if_prefix'),
])
def test_confirmed_cn_patterns(label, expected):
    assert rdp.matched_confirmed_category(label) == expected


@pytest.mark.parametrize('label,expected', [
    ('R10(2.2K)', 'r_paren_suffix'),
    ('R1(2.2K)', 'r_paren_suffix'),
    ('R1000(2.2K)', 'r_paren_suffix'),
    ('R10()', 'r_paren_suffix'),              # 括弧の中身は任意（空文字も可）
    ('VR5(10K)', 'vr_paren_suffix'),
    ('VR100(10K)', 'vr_paren_suffix'),
])
def test_confirmed_paren_suffix_patterns_use_full_label(label, expected):
    """R(...)/VR(...) は括弧の中身自体を問うため、`_judgment_text()` で
    括弧を取り除いた文字列ではなく正規化済みラベル全体（括弧含む）に対して
    判定する（basis='full'）。"""
    assert rdp.matched_confirmed_category(label) == expected


def test_confirmed_paren_suffix_requires_parens():
    """R10/VR5 単体（括弧なし）は r_paren_suffix/vr_paren_suffix の対象外。
    R10 は既存の letters_digits_2or3 で確定するが、VR5 はどの判定基準にも
    一致せず未確定のまま。"""
    assert rdp.matched_confirmed_category('R10') == 'letters_digits_2or3'
    assert rdp.matched_confirmed_category('VR5') is None


def test_is_confirmed_designator_requires_candidate_first():
    """is_confirmed_designator は機器符号候補であることが前提。除外パターン
    該当のラベルは、たとえ確定パターンの正規表現に文字列として一致しても
    確定扱いにしない。"""
    assert rdp.is_confirmed_designator('R10') is True
    assert rdp.is_confirmed_designator('FB') is False   # 候補だが確定でない
    assert rdp.is_confirmed_designator('GND') is False   # 除外パターン該当


# ---------------------------------------------------------------------------
# normalize_label / split_candidates
# ---------------------------------------------------------------------------

def test_normalize_label_nfkc_and_strip():
    assert rdp.normalize_label('　ＣＮ１　') == 'CN1'
    assert rdp.normalize_label('') == ''
    assert rdp.normalize_label(None) == ''


def test_split_candidates():
    """候補（＝除外パターンに該当しないもの）は Patterns 一致・除外非該当のみ。
    除外パターン該当（GND・TITLE）は結果に含まれない
    （2026-07-10 ユーザー指摘: RemainingUnclassified シートの中身を実データで
    再確認し、GND/TITLE等は除外カテゴリ付与済み＝候補に含まれないことを確定）。"""
    candidates = rdp.split_candidates(['R10', 'GND', 'CN3', 'TITLE'])
    assert candidates == ['R10', 'CN3']


def test_split_candidates_drops_non_pattern_matches():
    """3パターン（Patterns シート）のいずれにも一致しない文字列（説明文・記号等）も
    結果に含まれない。"""
    candidates = rdp.split_candidates(
        ['R10', 'GND', '(2/5)', '(-039-01)2/3', 'これは説明文'])
    assert candidates == ['R10']


def test_summarize_labels_counts_and_sorts():
    result = rdp.summarize_labels(['R10', 'R10', 'CN3'])
    assert result == [('CN3', 1), ('R10', 2)]


# ---------------------------------------------------------------------------
# 追加パターン（reference_deginator_pattern_added.txt、2026-07-26 追加確定）
# ---------------------------------------------------------------------------
#
# 優先順位: 候補形ゲート → 既存除外（judgment基準）→ 追加機器符号
# （ADDED_DESIGNATOR_PATTERNS、ラベル全体基準）→ 追加除外
# （ADDED_EXCLUSION_PATTERNS、ラベル全体基準）。
# 既存除外が「英単語」を優先的に除外し（MOTOR等）、追加機器符号は
# 追加除外より優先される（DC12A3等）。model/ref_designator_patterns.py の
# 「2b. 追加パターン」節冒頭コメント参照。

def test_compile_shorthand_pattern_notation():
    """独自記法（a=英大文字1字, n=数字1字, *=1回以上繰返し, .*=任意0文字以上）
    の変換が正しいことを直接検証する。"""
    rx = rdp._compile_shorthand_pattern('DCnnan')
    assert rx.match('DC12A3')
    assert not rx.match('DC1A3')     # 数字が1桁少ない
    assert not rx.match('DC123A34')  # 長さ不一致

    rx2 = rdp._compile_shorthand_pattern('Kn')
    assert rx2.match('K1')
    assert not rx2.match('K12')      # 'n' に * が無いので数字は1桁のみ

    rx3 = rdp._compile_shorthand_pattern('SW.*')
    assert rx3.match('SW')
    assert rx3.match('SW1')
    assert rx3.match('SW(ABC)')      # .* は括弧を含む


@pytest.mark.parametrize('label,expected_category', [
    ('MOT1', 'added_desig:MOT.*'),
    ('PG1', 'added_desig:PG.*'),
    ('SW1', 'added_desig:SW.*'),
    ('CB001', 'letters_digits_2or3'),      # 既存確定パターンが先に一致
    ('LS1', 'added_desig:LS.*'),
    ('OS1', 'added_desig:OS.*'),
    ('DC12A3', 'added_desig:DCnnan'),      # A衝突: 追加機器符号が追加除外に優先
])
def test_added_designator_patterns_confirmed(label, expected_category):
    """追加の機器符号パターンに一致するラベルは候補かつ確定（自動採用）になる。"""
    assert rdp.is_ref_designator_candidate(label) is True
    assert rdp.matched_confirmed_category(label) == expected_category


@pytest.mark.parametrize('label,expected_category', [
    ('DC12', 'added_excl:DCn*.*'),   # DCnnan(機器符号)には一致しないため除外のまま
    ('K1', 'added_excl:Kn'),
    ('O5', 'added_excl:O.*'),
    ('I9', 'added_excl:I.*'),
    ('CM100', 'added_excl:CM.*'),
])
def test_added_exclusion_patterns_excluded(label, expected_category):
    """追加の除外パターンに一致し、追加の機器符号パターンで救済されない
    ラベルは除外される。"""
    assert rdp.is_ref_designator_candidate(label) is False
    status, category = rdp.classify_judgment_detailed(
        rdp._judgment_text(label), label)
    assert status == 'excluded'
    assert category == expected_category


@pytest.mark.parametrize('label', ['MOTOR', 'PGND', 'SWITCH', 'CONTROL', 'THERMOCOUPLE'])
def test_existing_exclusion_wins_over_added_designator_prefix(label):
    """B衝突: 既存の除外語リスト（普通名詞・回路説明語）に完全一致する語は、
    その語幹が追加の機器符号パターン（MOT.*/PG.*/SW.*/CON.*/TH.* 等）に
    一致していても除外が優先される（英単語はそのまま除外、MOT1等の記号形
    だけ追加機器符号として救済する、2026-07-26 ユーザー確定）。"""
    assert rdp.is_ref_designator_candidate(label) is False


def test_added_patterns_do_not_affect_untouched_existing_labels():
    """追加パターンに一切関与しない既存ラベルの判定は変化しない（無回帰）。"""
    assert rdp.is_ref_designator_candidate('R10') is True
    assert rdp.matched_confirmed_category('R10') == 'letters_digits_2or3'
    assert rdp.is_ref_designator_candidate('GND') is False
    assert rdp.matched_confirmed_category('CN3') == 'cn_single_digit'
