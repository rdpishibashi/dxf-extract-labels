"""機器符号（候補）抽出（utils/ref_designator.py）の回帰テスト。

検証済み仕様（2026-07-10 確定、reference_designator_candidates.xlsx が正）:
  - パターン: 英字繰返し(-英字繰返し)?+数字繰返し+英数字/ハイフン任意、または英字繰返しのみ
  - 判定は括弧より前の部分に対して行う（R10(2.2K) → R10 で判定）
  - 除外: 単一英大文字・末尾+-・AWG・RACK*・図番・JIS/DWG*・A/B+数字・PE+数字・
    L/N/P+数字・X+英字、および普通名詞/回路説明/ユニット名/ケーブル色/図面情報枠語の
    完全一致リスト
  - 図面枠 = lineweight=frame_lineweight かつ color=7（region_detector と同条件。
    lineweight単独では無関係な線分を拾い検出が壊れることを実データで確認済み）
  - 図面情報欄・枠外位置記号は、図面枠線を直接の子に持つ「フォーマットブロック」
    （実データでは JZB_*）の INSERT 由来であることを利用して構造的に除外する
    （人名リストは持たない）

実 DXF（プロジェクト直下のサンプル）を使ったブラックボックステストも含む。
サンプルが無い環境ではスキップする。
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from utils import ref_designator  # noqa: E402

SAMPLE_DIR = os.path.join(PROJECT_ROOT, 'sample-dxf')


def _find_sample(name):
    direct = os.path.join(SAMPLE_DIR, name)
    if os.path.exists(direct):
        return direct
    for root, _dirs, files in os.walk(SAMPLE_DIR):
        if name in files:
            return os.path.join(root, name)
    return None


# ---------------------------------------------------------------------------
# is_ref_designator_candidate — パターン一致
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('label', [
    'R10', 'CN3', 'D4', 'CN1',                   # 英字+数字（A/Bは機器端子として除外対象）
    'FB', 'CNCNT', 'MSS',                        # 英字のみ
    'CN-IF2-1', 'AAC1B4-07', 'ETC-JP2',           # 英字-英字+数字(+ハイフン任意)
    'R10(2.2K)', 'FB()', 'MSS(MOTOR)', 'U23B(DAC)',  # 括弧付き（括弧前で判定）
])
def test_candidate_pattern_matches(label):
    assert ref_designator.is_ref_designator_candidate(label) is True


@pytest.mark.parametrize('label', [
    '123',        # 数字のみ（英字繰返しが先頭に必須）
    '',           # 空文字列
    '(BU)',       # 括弧が先頭（判定対象が空になる）
    'R 10',       # スペースを含む
    'あいう',      # 英数字以外
])
def test_candidate_pattern_rejects(label):
    assert ref_designator.is_ref_designator_candidate(label) is False


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
])
def test_exclusion_categories(label, reason):
    assert ref_designator.is_ref_designator_candidate(label) is False, (
        f'{label} should be excluded ({reason})')


def test_x_plus_digit_is_not_excluded():
    """X+数字（X1, X24）はIEC上の正規の端子/コネクタ記号であり、
    X+英字（io_signal_x_prefix）とは区別して候補に残す。"""
    assert ref_designator.is_ref_designator_candidate('X1') is True
    assert ref_designator.is_ref_designator_candidate('X24') is True


@pytest.mark.parametrize('label', ['OUT2', 'IN1', 'COM3', 'GND9', 'POWER5'])
def test_circuit_description_excludes_keyword_plus_single_digit(label):
    """circuit_description のキーワードは末尾に数字1桁があっても除外対象
    （例 OUT2, IN1, COM3。2026-07-10 ユーザー指摘）。"""
    assert ref_designator.is_ref_designator_candidate(label) is False
    assert ref_designator.classify_judgment_detailed(label) == ('excluded', 'circuit_description')


@pytest.mark.parametrize('label', ['OUT12', 'IN10', 'GND12'])
def test_circuit_description_two_digit_suffix_not_excluded(label):
    """数字2桁以上は circuit_description の除外対象外（候補として残る）。"""
    status, _category = ref_designator.classify_judgment_detailed(label)
    assert status != 'excluded'


def test_person_name_not_excluded_by_content_filter():
    """人名リストは実装に持たない（図面情報枠の構造的除外で対応する設計）ため、
    単体の is_ref_designator_candidate は人名を弾かない。"""
    assert ref_designator.is_ref_designator_candidate('KURIHARA') is True


# ---------------------------------------------------------------------------
# 確定パターン（CONFIRMED_PATTERN_CATEGORIES / matched_confirmed_category /
# is_confirmed_designator / split_confirmed）— 未確定ラベルUIのレビュー不要判定
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('label,expected', [
    ('R10', 'letters_digits_2or3'),
    ('R100', 'letters_digits_2or3'),
    ('CN3', None),                        # 数字1桁・複数文字プレフィックスは不一致
    ('R10A', 'letters_digits_2or3_letter'),
    ('CN-IF21', 'hyphen_letters_digits_notail'),
    ('CN-IF2-1', None),                   # 末尾に続きがあるので対象外
    ('D1', 'single_letter_digits_except_ab'),
    ('D1000', 'single_letter_digits_except_ab'),   # 桁数不問
    ('A1', None),                         # A,Bは対象外（単一英字パターンのみ）
    ('B12', 'letters_digits_2or3'),       # A,B除外は単一英字パターン限定
])
def test_matched_confirmed_category(label, expected):
    assert ref_designator.matched_confirmed_category(label) == expected


def test_is_confirmed_designator_requires_candidate_first():
    """is_confirmed_designator は機器符号候補であることが前提。除外パターン
    該当のラベルは、たとえ確定パターンの正規表現に文字列として一致しても
    確定扱いにしない。"""
    assert ref_designator.is_confirmed_designator('R10') is True
    assert ref_designator.is_confirmed_designator('CN3') is False   # 候補だが確定でない
    assert ref_designator.is_confirmed_designator('GND') is False   # 除外パターン該当


def test_split_confirmed():
    labels = [('R10', 0, 0), ('CN3', 1, 1), ('D100', 2, 2)]
    confirmed, review = ref_designator.split_confirmed(labels)
    assert confirmed == [('R10', 0, 0), ('D100', 2, 2)]
    assert review == [('CN3', 1, 1)]


# ---------------------------------------------------------------------------
# normalize_label / split_candidates
# ---------------------------------------------------------------------------

def test_normalize_label_nfkc_and_strip():
    assert ref_designator.normalize_label('　ＣＮ１　') == 'CN1'
    assert ref_designator.normalize_label('') == ''
    assert ref_designator.normalize_label(None) == ''


def test_split_candidates():
    """候補（＝未確定ラベルUIでのレビュー対象）は Patterns 一致・除外非該当のみ。
    除外パターン該当（GND・TITLE）は結果に含まれない
    （2026-07-10 ユーザー指摘: RemainingUnclassified シートの中身を実データで
    再確認し、GND/TITLE等は除外カテゴリ付与済み＝候補に含まれないことを確定）。"""
    candidates = ref_designator.split_candidates(['R10', 'GND', 'CN3', 'TITLE'])
    assert candidates == ['R10', 'CN3']


def test_split_candidates_drops_non_pattern_matches():
    """3パターン（Patterns シート）のいずれにも一致しない文字列（説明文・記号等）も
    結果に含まれない。"""
    candidates = ref_designator.split_candidates(
        ['R10', 'GND', '(2/5)', '(-039-01)2/3', 'これは説明文'])
    assert candidates == ['R10']


def test_summarize_labels_counts_and_sorts():
    result = ref_designator.summarize_labels(['R10', 'R10', 'CN3'])
    assert result == [('CN3', 1), ('R10', 2)]


# ---------------------------------------------------------------------------
# build_labeled_rows
# ---------------------------------------------------------------------------

def test_build_labeled_rows_without_region():
    rows = ref_designator.build_labeled_rows([('R10', 0, 0), ('R10', 10, 10), ('CN3', 5, 5)])
    assert rows == [{'ラベル': 'CN3', '個数': 1}, {'ラベル': 'R10', '個数': 2}]


def test_build_labeled_rows_with_region():
    square = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    named = [{'polygon': square, 'name': 'RACK1', 'id': 0, 'frame': 0, 'area_pct': 50.0}]
    rows = ref_designator.build_labeled_rows(
        [('R10', 10, 10), ('CN3', 200, 200)], named_regions=named)
    by_label = {r['ラベル']: r for r in rows}
    assert by_label['R10']['領域'] == 'RACK1'
    assert by_label['CN3']['領域'] == ''


# ---------------------------------------------------------------------------
# build_named_regions / build_region_output（合成データ）
# ---------------------------------------------------------------------------

def test_build_named_regions_auto_names_unnamed():
    """名称候補が無い領域は自動命名される。候補がある領域はユーザー選択
    （name_selections）が無い限り named に含まれない（build_region_results と同じ挙動）。"""
    analysis = {
        'regions': [
            {'id': 0, 'frame': 0, 'polygon': [], 'area_pct': 10.0, 'name_candidates': []},
            {'id': 1, 'frame': 0, 'polygon': [], 'area_pct': 10.0,
             'name_candidates': [(1.0, 'ＲＡＣＫ１')]},
        ]
    }
    named, next_idx = ref_designator.build_named_regions(analysis, {}, 'a.dxf')
    names = sorted(r['name'] for r in named)
    assert names == ['no name 1']   # 候補ありの領域1は未選択のため named に含まれない
    assert next_idx == 1

    # 明示的に選択すれば全角→半角正規化された名称が named に入る
    selections = {('a.dxf', 1): ['ＲＡＣＫ１']}
    named2, _ = ref_designator.build_named_regions(analysis, selections, 'a.dxf')
    names2 = sorted(r['name'] for r in named2)
    assert names2 == ['RACK1', 'no name 1']


def test_build_region_output_counts_and_sorts():
    square = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    named = [{'polygon': square, 'name': 'RACK1', 'id': 0, 'frame': 0, 'area_pct': 50.0}]
    labels = [('R10', 10, 10), ('R10', 20, 20), ('CN3', 500, 500)]
    out = ref_designator.build_region_output(labels, named, sort_value='desc')
    assert [r['ラベル'] for r in out['rows']] == ['R10', 'CN3']   # desc
    by_label = {r['ラベル']: r for r in out['rows']}
    assert by_label['R10']['個数'] == 2
    assert by_label['R10']['領域'] == 'RACK1'
    assert by_label['CN3']['領域'] == ''
    assert out['in_region_count'] == 2
    assert out['region_label_counts'] == {'RACK1': {'R10': 2}}
    assert named[0]['label_count'] == 2   # named に label_count が付与される


# ---------------------------------------------------------------------------
# 図面枠検出・フォーマットブロック除外（実 DXF）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sample_name', [
    'EE6868-500-01C.dxf',   # 13図面（複数図面ファイル）
    'EE6097-039-06C.dxf',   # 3図面
])
def test_titleblock_content_excluded_from_real_sample(sample_name):
    """図面情報欄内のラベル（TITLE/DATE/REVISION/人名等）が機器符号候補
    （確定＋未確定ラベルUIでのレビュー対象）に一切現れないこと（構造的除外の検証）。"""
    path = _find_sample(sample_name)
    if not path:
        pytest.skip(f'sample DXF not found: {sample_name}')

    data = ref_designator.extract_ref_designator_data(
        path, frame_lineweight=100, original_filename=sample_name)
    assert data['warning'] is None
    all_texts = {t for t, _x, _y in data['confirmed_labels']} | \
        {t for t, _x, _y in data['review_labels']}

    titleblock_leakage = {'TITLE', 'DATE', 'REVISION', 'APPRV', 'CHECK', 'DESIG',
                           'DRAW', 'MARK', 'REMARKS', 'SCALE', 'NAME',
                           'KURIHARA', 'MORIOKA', 'KANAI'} & all_texts
    assert not titleblock_leakage, f'titleblock content leaked: {titleblock_leakage}'
    assert data['total_in_frame'] > 0


@pytest.mark.parametrize('sample_name', [
    'EE6868-500-01C.dxf',
    'EE6097-039-06C.dxf',
])
def test_candidate_labels_all_match_base_pattern_and_exclude_known_words(sample_name):
    """機器符号（候補）＝確定＋未確定ラベルUIでのレビュー対象は (1) Patterns（3パターン）に
    一致したものだけで (2) 除外パターン該当（GND・INPUT・TITLE等）は一切含まれず、
    (3) 確定と未確定は互いに排他であること（2026-07-10 ユーザー指摘のバグ修正の
    回帰検証。実データで RemainingUnclassified シートの中身を再確認し、
    GND/INPUT/TITLE等には除外カテゴリが付与されている＝候補に含まれないのが
    正しい動作と確定）。"""
    path = _find_sample(sample_name)
    if not path:
        pytest.skip(f'sample DXF not found: {sample_name}')

    data = ref_designator.extract_ref_designator_data(
        path, frame_lineweight=100, original_filename=sample_name)
    excluded_words = {'GND', 'INPUT', 'OUTPUT', 'TITLE', 'POWER', 'ALARM'}
    confirmed_texts = {t for t, _x, _y in data['confirmed_labels']}
    review_texts = {t for t, _x, _y in data['review_labels']}
    assert confirmed_texts.isdisjoint(review_texts)
    for text, _x, _y in list(data['confirmed_labels']) + list(data['review_labels']):
        judgment = ref_designator._judgment_text(text)
        assert ref_designator.CANDIDATE_PATTERN.match(judgment), (
            f'{text!r} does not match CANDIDATE_PATTERN')
        assert text not in excluded_words, (
            f'{text!r} matches ExclusionPatterns but leaked into candidate pool')
    for text in confirmed_texts:
        assert ref_designator.matched_confirmed_category(
            ref_designator._judgment_text(text)) is not None


def test_collect_in_frame_labels_frame_count_matches_known_drawing_count():
    """EE6868-500-01C.dxf は13図面から構成される（スキル文書で確認済みの既知事実）。
    lineweight+color=7 併用で正しく13枠を検出できることを回帰検証する
    （lineweight単独だと772本の無関係線分を拾い誤検出31枚になることを
    実装時に確認済み、2026-07-10）。"""
    path = _find_sample('EE6868-500-01C.dxf')
    if not path:
        pytest.skip('sample DXF not found: EE6868-500-01C.dxf')
    result = ref_designator.collect_in_frame_labels(path, frame_lineweight=100)
    assert result['error'] is None
    assert len(result['frames']) == 13


def test_extract_ref_designator_rows_end_to_end():
    """通常モード用トップレベル関数が実DXFに対し例外なく行データを返す。"""
    path = _find_sample('EE6491-039-04A.dxf')
    if not path:
        pytest.skip('sample DXF not found: EE6491-039-04A.dxf')
    result = ref_designator.extract_ref_designator_rows(
        path, frame_lineweight=100, original_filename='EE6491-039-04A.dxf')
    assert result['filename'] == 'EE6491-039-04A.dxf'
    assert result['warning'] is None
    assert len(result['confirmed_rows']) > 0
    assert len(result['review_rows']) > 0
    # 行データは {'ラベル': str, '個数': int} の形
    for sample in (result['confirmed_rows'][0], result['review_rows'][0]):
        assert set(sample.keys()) == {'ラベル', '個数'}
