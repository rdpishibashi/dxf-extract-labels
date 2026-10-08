"""機器符号抽出パイプライン（model/ref_designator.py）の回帰テスト。

**2026-07-30 判定条件の簡素化（ユーザー指定）**: 以下のパイプラインに一本化した。

  1. 図面枠内、かつ図面情報枠（タイトルブロック）外のラベルをスコープとする
  2. ラベル末尾に括弧で閉じた文字列がある場合は、その括弧より前の部分で
     判定する（出力は原文のまま）
  3. 3つの候補パターン（英字繰返し(-英字繰返し)?+数字繰返し+英数字/ハイフン
     任意、または英字繰返しのみ）のいずれかに一致するラベルを、除外や
     確認なしにそのまま全件出力する

除外パターン・追加パターン・確定パターン・「未確定ラベル」UIでのレビュー・
連動採用・判断ログは廃止した。これらのパターン判定ロジック自体は
`region_detector.py` の領域名候補フィルタが引き続き利用しているため
`ref_designator_patterns.py` に残っており、対応する回帰テストは
`tests/regression/test_ref_designator_patterns.py` にある。

図面情報枠（タイトルブロック）の除外は、旧来の構造的除外（図面枠線を
直接の子に持つ「フォーマットブロック」由来のテキストを丸ごと除外）から、
幾何検出（図面枠の右下、右辺・底辺が図面枠に接する矩形をタイトル項目名の
並びから特定する）に置き換えた。詳細は `_detect_info_frame_bbox()` の
docstring、および比較検証（旧方式 vs 新方式、実データ156件、
`sample-dxf/` 配下）の結果を参照。

図面枠 = lineweight=frame_lineweight かつ color=7（region_detector と同条件。
lineweight単独では無関係な線分を拾い検出が壊れることを実データで確認済み）。

実 DXF（プロジェクト直下のサンプル）を使ったブラックボックステストも含む。
サンプルが無い環境ではスキップする。
"""
import os
import sys
import tempfile

import ezdxf
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model import ref_designator  # noqa: E402
from model.region_detector import _collect_region_geometry, DEFAULT_REGION_CONFIG  # noqa: E402
from model.terminal_detector import _collect_geometry as _collect_terminal_geometry  # noqa: E402
from model.extract_labels import extract_labels  # noqa: E402

SAMPLE_DIR = os.path.join(PROJECT_ROOT, 'sample-dxf')


def _find_sample(name):
    direct = os.path.join(SAMPLE_DIR, name)
    if os.path.exists(direct):
        return direct
    # curated フォルダを固定順で先に探す（2026-08 追加）。
    # 下の os.walk による再帰検索は到達順が保証されず、pairC（同一図番の
    # 比較元/比較先が同名・別内容で並ぶペアデータ）を先に掴むと、テストが
    # 意図しない版のDXFを読んでしまう。実際に EE6868-500-01C.dxf 等
    # 9ファイルが該当していた（pairC 側は設計上同名で正しいため、
    # データではなく探索順で解決する）。
    for _sub in ('problems', 'samples', 'terminal-detector', 'viewer-error'):
        _p = os.path.join(SAMPLE_DIR, _sub, name)
        if os.path.exists(_p):
            return _p
    for root, _dirs, files in os.walk(SAMPLE_DIR):
        if name in files:
            return os.path.join(root, name)
    return None


# ---------------------------------------------------------------------------
# classify_labels — 3候補パターンに一致すれば全件そのまま出力（除外なし）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('label', [
    'R10', 'CN3', 'D4', 'CN1', 'A1', 'B12',       # 英字+数字（除外なしなのでA/Bも通る）
    'FB', 'CNCNT', 'MSS',                          # 英字のみ
    'CN-IF2-1', 'AAC1B4-07', 'AB-CD12',            # 英字-英字+数字(+ハイフン任意)
    'R10(2.2K)', 'FB()', 'MSS(MOTOR)', 'U23B(DAC)',  # 括弧付き（括弧前で判定）
    'GND', 'POWER', 'MOTOR', 'TITLE', 'DATE',      # 旧除外パターン該当語（新仕様では候補になる）
    'AWG14', 'RACK1', 'W1234', 'XRST',             # 旧除外パターン該当語（同上）
])
def test_classify_labels_keeps_all_pattern_matches(label):
    """3候補パターンに一致するものはすべて出力される（除外パターンは廃止）。"""
    labels = [(label, 0.0, 0.0)]
    result = ref_designator.classify_labels(labels)
    assert result == labels


@pytest.mark.parametrize('label', [
    '123',        # 数字のみ（英字繰返しが先頭に必須）
    '',           # 空文字列
    '(BU)',       # 括弧が先頭（判定対象が空になる）
    'R 10',       # スペースを含む
    'あいう',      # 英数字以外
])
def test_classify_labels_rejects_non_pattern_matches(label):
    labels = [(label, 0.0, 0.0)] if label else []
    result = ref_designator.classify_labels(labels)
    assert result == []


def test_classify_labels_judges_before_parenthesis_but_outputs_original():
    """括弧より前の部分で判定し、出力は原文のまま（例: 'R10(1234)' は
    判定用文字列 'R10' がパターンに一致するため候補になり、出力は
    括弧付きの原文 'R10(1234)' のまま）。"""
    labels = [('R10(1234)', 1.0, 2.0)]
    assert ref_designator.classify_labels(labels) == labels


@pytest.mark.parametrize('label', [
    # 括弧の直前にスペースがある実データ表記（2026-09-11、ユーザー提供の
    # extracted_labels.xlsx「抽出ラベル一覧」シートで、機器符号候補列が空欄に
    # なっていた94件中73件がこの形。「機器符号候補のはずなのに一致しない」と
    # いう報告を受けて発覚した不具合の修正）。
    'CB002 (15A)',
    'Q10 (Q2)',
    'TMP (TMP-1003LM)',
    'MC001 (-000-05)2/2',
    # ' **' 区切り（2026-09-11 ユーザー指示。実データ表記
    # '型式 ** (正式な型番)'）。
    'FL1F1 ** (FL1F-H12RCE)',
    'SX03 ** (NX-SID800)',
    # 'W No.' 前方一致（2026-09-11 ユーザー指示）。実データでピリオド直後の
    # スペース有無・末尾説明文・数字4桁連続を含む型番など表記ゆれがある。
    'W No.M06004',
    'W No. C35019',
    'W No.M06008(CC050VZRB)',
    'W No.M39300 ACCESSORY CABLE (MR-J3ENCBL10M-A1-H)',
])
def test_is_ref_designator_label_2026_09_11_fixes(label):
    """機器符号候補として扱われるべき実データ表記（括弧前の空白除去・' **'区切り・
    'W No.'前方一致）が、主抽出パイプラインの `is_ref_designator_label()`
    （「機器符号候補」列の判定に使う関数、除外パターン不使用）で正しく候補と
    判定されることを固定する。"""
    assert ref_designator.is_ref_designator_label(label) is True


def test_is_ref_designator_label_w_no_prefix_rejects_non_matching_casing():
    """'W No.' の大文字小文字・スペースの有無を区別する（実データの文中注記
    'Wno.M39301' 等〈ラベルではなく説明文中の言及〉を誤って候補にしないため）。"""
    assert ref_designator.is_ref_designator_label('Wno.M39301') is False
    assert ref_designator.is_ref_designator_label('w no.M06004') is False


@pytest.mark.parametrize('label', [
    # Ω（オーム記号）を含む語の直前で区切る（2026-09-11 ユーザー指示）。
    # ユーザー提供Excelで「機器符号候補のはず」と報告された括弧なし表記。
    'R0 2.2KΩ',
    'R1 2.2KΩ',
    'R84 4.7KΩ',
])
def test_is_ref_designator_label_ohm_symbol_fix(label):
    """括弧を使わずスペース区切りで抵抗値（Ω）が続く実データ表記が、主抽出
    パイプラインの `is_ref_designator_label()` で正しく候補と判定される
    ことを固定する。"""
    assert ref_designator.is_ref_designator_label(label) is True


@pytest.mark.parametrize('label', [
    'H', 'L', 'S', 'V', 'A', 'B', 'C', 'X', 'R', 'Z',
    'H(5A)',  # 括弧を除いた判定用文字列が単一英字になるケースも対象
])
def test_is_ref_designator_label_excludes_single_uppercase_letter(label):
    """英大文字1字だけの文字列（図形枠外の位置記号）は機器符号候補ではない
    （2026-09-14ユーザー指摘で発覚した回帰の修正）。

    `ref_designator_patterns.EXCLUSION_REGEX_CATEGORIES` の
    `single_letter_position`（`^[A-Z]$`）は元々この除外を担っていたが、
    v2.0.0で`is_ref_designator_label()`が除外パターンを一切使わない方式に
    簡素化された際、この1件だけ意図せず失われていた
    （`letters_only`パターン`^[A-Z]+$`が単一英字も候補として拾ってしまう
    ため）。`region_detector.py`の領域名候補フィルタ
    （`classify_judgment_detailed()`経由、こちらは除外パターンを使う旧方式
    のまま）は元々正しく除外していたため無関係——今回追加したのは
    `is_ref_designator_label()`専用の例外（`_SINGLE_LETTER_PATTERN`）。"""
    assert ref_designator.is_ref_designator_label(label) is False


@pytest.mark.parametrize('label', ['AB', 'FB', 'CNCNT', 'R1', 'C1'])
def test_is_ref_designator_label_still_accepts_multi_char_labels(label):
    """単一英字の除外追加が、2文字以上の英字繰返し（`AB`等）・英字+数字
    （`R1`等）を誤って除外していないことを確認する（回帰防止）。"""
    assert ref_designator.is_ref_designator_label(label) is True


def test_region_name_candidate_filter_unaffected_by_single_letter_fix():
    """`is_ref_designator_label()`への単一英字除外の追加が、
    `region_detector.py`の領域名候補フィルタが使う
    `classify_judgment_detailed()`（`PATTERN_CATEGORIES`を直接参照する別経路）
    には影響しないことを固定する（`_SINGLE_LETTER_PATTERN`は
    `is_ref_designator_label()`内のみのローカルな例外であり、
    `ref_designator_patterns.PATTERN_CATEGORIES`自体は変更していないため）。"""
    from model.ref_designator_patterns import classify_judgment_detailed
    status, cat = classify_judgment_detailed('H', 'H')
    assert (status, cat) == ('excluded', 'single_letter_position')


@pytest.mark.parametrize('label', ['N24-', 'CN1-', 'R10-', 'L1+', 'V+', 'V-'])
def test_is_ref_designator_label_excludes_trailing_sign(label):
    """末尾が "+"/"-" で終わる文字列（電源端子表記、例 `N24-`・`L1+`）は
    機器符号候補ではない（2026-09-14ユーザー指摘で発覚した回帰の修正）。

    `ref_designator_patterns.EXCLUSION_REGEX_CATEGORIES` の
    `trailing_sign`（`.*[+-]$`）は元々この除外を担っていたが、
    v2.0.0で`is_ref_designator_label()`が除外パターンを一切使わない方式に
    簡素化された際、この1件だけ意図せず失われていた（`letters_digits_any`
    〈`^[A-Z]+[0-9]+[A-Z0-9-]*$`〉の末尾`[A-Z0-9-]*`が末尾ハイフンを許容
    してしまうため、`N24-`・`CN1-`等が誤って機器符号候補='Y'になっていた。
    末尾"+"は元々どの候補パターンにも"+"という文字自体が含まれないため
    既に候補にならないが、`_TRAILING_SIGN_PATTERN`は`trailing_sign`と同じ
    `[+-]`両対応で実装し、将来パターンに"+"が追加されても安全にしてある）。

    単一英字の除外（`_SINGLE_LETTER_PATTERN`）とは異なり
    `EXCLUSION_EXACT_CATEGORIES`（普通名詞・回路説明語・ケーブル色・
    図面情報枠用語・ユニット名。GND・SYSTEM・MOTOR等）は**復元していない**
    （2026-09-14ユーザー確認。v2.0.0でこれらを候補に含めるよう明示的に
    方針変更した記録がVERSION_HISTORY.mdにあるため、単一英字・末尾+/-とは
    扱いが異なる）。"""
    assert ref_designator.is_ref_designator_label(label) is False


@pytest.mark.parametrize('label', ['GND', 'SYSTEM', 'MOTOR', 'OVERHEAT', 'WHITE', 'RED'])
def test_is_ref_designator_label_still_accepts_common_nouns(label):
    """普通名詞・回路説明語・ケーブル色（`GND`・`SYSTEM`・`MOTOR`・`WHITE`等）は
    引き続き機器符号候補として扱われることを固定する（v2.0.0の明示的な方針
    〈EXCLUSION_EXACT_CATEGORIESを判定に使わない〉を維持している回帰防止、
    2026-09-14ユーザー確認）。"""
    assert ref_designator.is_ref_designator_label(label) is True


# ---------------------------------------------------------------------------
# normalize_labels
# ---------------------------------------------------------------------------

def test_normalize_labels_nfkc_and_strip():
    labels = [('　ＣＮ１　', 0.0, 0.0), ('', 1.0, 1.0), ('R10', 2.0, 2.0)]
    result = ref_designator.normalize_labels(labels)
    assert result == [('CN1', 0.0, 0.0), ('R10', 2.0, 2.0)]


# ---------------------------------------------------------------------------
# build_labeled_rows
# ---------------------------------------------------------------------------

def test_build_labeled_rows_without_region():
    """'機器符号候補' 列は R10/CN3 のようなパターン一致ラベルで 'CAN' になる
    （v3.0.0で全ラベルを対象にするようになったため追加された列。v3.10.0で
    機器符号一覧に一致する場合は 'DEF' になるよう3値化されたが、R10/CN3は
    一覧に無いため master_index=None 時と同じ 'CAN' のままで変わらない）。"""
    rows = ref_designator.build_labeled_rows([('R10', 0, 0), ('R10', 10, 10), ('CN3', 5, 5)])
    assert rows == [
        {'機器符号候補': 'CAN', 'ラベル': 'CN3', '個数': 1},
        {'機器符号候補': 'CAN', 'ラベル': 'R10', '個数': 2},
    ]


def test_build_labeled_rows_marks_non_matching_label_as_none():
    """パターンに一致しないラベル（スペース入り等）は '機器符号候補' が None。"""
    rows = ref_designator.build_labeled_rows([('R10', 0, 0), ('NOTE ONE', 5, 5)])
    by_label = {r['ラベル']: r['機器符号候補'] for r in rows}
    # R10は機器符号一覧に無いため、v3.10.0でも'CAN'（候補パターン一致のみ）
    assert by_label['R10'] == 'CAN'
    assert by_label['NOTE ONE'] is None


def test_build_labeled_rows_with_region():
    square = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    named = [{'polygon': square, 'name': 'RACK1', 'id': 0, 'frame': 0, 'area_pct': 50.0}]
    rows = ref_designator.build_labeled_rows(
        [('R10', 10, 10), ('CN3', 200, 200)], named_regions=named)
    by_label = {r['ラベル']: r for r in rows}
    assert by_label['R10']['領域'] == 'RACK1'
    assert by_label['CN3']['領域'] == ''
    # R10は機器符号一覧（master_index）に無いため、v3.10.0ではCAN
    # （候補パターン一致のみ。master_index=None時のDEF/CAN判定は
    # test_designator_def_can.py が検証する）
    assert by_label['R10']['機器符号候補'] == 'CAN'


# ---------------------------------------------------------------------------
# build_named_regions / build_region_output（合成データ）
# ---------------------------------------------------------------------------

def test_build_named_regions_auto_names_unnamed():
    """名称候補が無い領域は自動命名される。候補がある領域はユーザー選択
    （name_selections）が無い限り named に含まれない。
    無名領域が1個だけの図面では番号を付けず「no name」のみにする（2026-07-23、
    ユーザー指摘: 番号は複数の無名領域を区別するためのものであり、1個しか
    無いのに番号が付くとその意味が伝わらず不安になるため）。"""
    analysis = {
        'regions': [
            {'id': 0, 'frame': 0, 'polygon': [], 'area_pct': 10.0, 'name_candidates': []},
            {'id': 1, 'frame': 0, 'polygon': [], 'area_pct': 10.0,
             'name_candidates': [(1.0, 'ＲＡＣＫ１')]},
        ]
    }
    named, next_idx = ref_designator.build_named_regions(analysis, {}, 'a.dxf')
    names = sorted(r['name'] for r in named)
    assert names == ['no name']   # 無名領域はこの図面に1個だけなので番号なし
    assert next_idx == 1

    # 明示的に選択すれば全角→半角正規化された名称が named に入る
    selections = {('a.dxf', 1): ['ＲＡＣＫ１']}
    named2, _ = ref_designator.build_named_regions(analysis, selections, 'a.dxf')
    names2 = sorted(r['name'] for r in named2)
    assert names2 == ['RACK1', 'no name']


def test_build_named_regions_numbers_multiple_unnamed():
    """同じ図面内に無名領域が2個以上ある場合は、従来通り連番で区別する
    （「no name 1」「no name 2」）。"""
    analysis = {
        'regions': [
            {'id': 0, 'frame': 0, 'polygon': [], 'area_pct': 10.0, 'name_candidates': []},
            {'id': 1, 'frame': 0, 'polygon': [], 'area_pct': 10.0, 'name_candidates': []},
        ]
    }
    named, next_idx = ref_designator.build_named_regions(analysis, {}, 'a.dxf')
    names = sorted(r['name'] for r in named)
    assert names == ['no name 1', 'no name 2']
    assert next_idx == 2


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
# build_ref_designator_final / build_ref_designator_region_results（合成データ）
# ---------------------------------------------------------------------------

def test_build_ref_designator_final():
    """'ref_designator_count'（v3.0.0新設）は '機器符号候補' が 'DEF'/'CAN'
    （v3.10.0で3値化。旧'Y'に相当）の行の個数合計。R10/CN3 はいずれも
    パターンに一致するため（かつ機器符号一覧には無いため'CAN'のまま）1+1=2。"""
    ref_data_by_file = {
        'a.dxf': {
            'labels': [('R10', 0, 0), ('CN3', 1, 1)],
            'total_in_frame': 5,
            'frames': 1,
            'warning': None,
            'main_drawing_number': 'EE1234-500-01A',
            'source_drawing_number': None,
            'title': 'TITLE-A',
            'subtitle': None,
        },
    }
    ref_final = ref_designator.build_ref_designator_final(ref_data_by_file, sort_value='asc')
    assert ref_final['a.dxf']['rows'] == [
        {'機器符号候補': 'CAN', 'ラベル': 'CN3', '個数': 1},
        {'機器符号候補': 'CAN', 'ラベル': 'R10', '個数': 1},
    ]
    assert ref_final['a.dxf']['total_in_frame'] == 5
    assert ref_final['a.dxf']['frames'] == 1
    assert ref_final['a.dxf']['ref_designator_count'] == 2
    assert ref_final['a.dxf']['main_drawing_number'] == 'EE1234-500-01A'
    assert ref_final['a.dxf']['title'] == 'TITLE-A'


def test_build_ref_designator_region_results():
    """'ref_designator_count'（v3.0.0新設）が 'final_count'/'filtered_count'
    （旧「機器符号（候補）以外も抽出」用、v3.0.0で廃止）に代わって提供される。
    'subtitle' も region_results に引き継がれる（v3.0.0、Summaryシートに
    サブタイトルが表示されないバグの修正）。

    'total_in_frame' は v3.11.0 で 'kept_labels'（除外後のラベル数）から
    再計算するようになった（領域除外機能の追加に伴う。除外指定が無い場合は
    常に len(data['labels']) と一致する——実際のパイプラインでも
    extract_ref_designator_data() は info['total_in_frame'] = len(normalized)
    と info['labels'] = normalized を同時に設定するため、
    'total_in_frame' が len(labels) と異なることは実データでは起こらない。
    旧テストはこの不変条件に反する人為的な値〈4〉を設定しており、
    「入力値をそのまま通す」という実装詳細を検証していただけだった）。"""
    square = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    ref_data_by_file = {
        'a.dxf': {'labels': [('R10', 10, 10), ('CN3', 200, 200)], 'total_in_frame': 2},
    }
    region_analyses = {
        'a.dxf': {
            'frames': [(0, 0, 0, 0)],
            'regions': [{'id': 0, 'frame': 0, 'polygon': square, 'area_pct': 50.0,
                         'name_candidates': [(1.0, 'RACK1')]}],
            'main_drawing_number': 'EE1234-500-01A',
            'title': 'TITLE-A',
            'subtitle': 'SUB-A',
        },
    }
    name_selections_by_file = {'a.dxf': {('a.dxf', 0): ['RACK1']}}
    results = ref_designator.build_ref_designator_region_results(
        ref_data_by_file, region_analyses, name_selections_by_file, sort_value='asc')
    r = results['a.dxf']
    assert r['total_in_frame'] == 2   # len(labels)。除外指定が無いため不変
    assert r['ref_designator_count'] == 2   # R10 + CN3、いずれもパターン一致
    assert r['subtitle'] == 'SUB-A'
    assert r['title'] == 'TITLE-A'
    assert r['drawing_number'] == 'EE1234-500-01A'
    by_label = {row['ラベル']: row for row in r['rows']}
    assert by_label['R10']['領域'] == 'RACK1'
    assert by_label['CN3']['領域'] == ''
    # R10は機器符号一覧（master_index）に無いため、v3.10.0ではCAN（上記参照）
    assert by_label['R10']['機器符号候補'] == 'CAN'
    # region_rows: 領域一覧用の全領域データ（v3.0.0新設）
    assert len(r['region_rows']) == 1
    assert r['region_rows'][0]['name'] == 'RACK1'
    assert r['region_rows'][0]['label_count'] == 1   # RACK1内はR10のみ
    assert r['region_rows'][0]['ref_designator_count'] == 1
    # mark: 除外指定なし・RACK1は特定済みなので'特定'（v3.11.0新設）
    assert r['region_rows'][0]['mark'] == '特定'


# ---------------------------------------------------------------------------
# build_all_regions_summary（v3.0.0新設: 「領域一覧」用、全領域を対象にする）
# ---------------------------------------------------------------------------

def test_build_all_regions_summary_includes_unconfirmed_region():
    """`build_named_regions()` は候補があるが未確定の領域を除外するが、
    `build_all_regions_summary()` は確定・未確定を問わずすべての領域を対象にする
    （v3.0.0、要求「検出結果にかかわらず検出した領域をすべて記載」）。"""
    square = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    analysis = {
        'regions': [
            {'id': 0, 'frame': 0, 'polygon': square, 'area_pct': 30.0,
             'name_candidates': [(1.0, 'FX CHAMBER'), (2.0, 'CM DRV')]},
        ]
    }
    # name_selections が空 = 「領域一覧」で何もチェックされておらず未確定。
    # build_named_regions() ではこの領域は named に含まれない。
    named, _ = ref_designator.build_named_regions(analysis, {}, 'a.dxf')
    assert named == []

    # build_all_regions_summary() は未確定でも候補をカンマ区切りで載せる。
    rows = ref_designator.build_all_regions_summary(analysis, {}, 'a.dxf', [])
    assert len(rows) == 1
    assert rows[0]['name'] == 'FX CHAMBER, CM DRV'


def test_build_all_regions_summary_confirmed_region_uses_chosen_name():
    square = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    analysis = {
        'regions': [
            {'id': 0, 'frame': 0, 'polygon': square, 'area_pct': 30.0,
             'name_candidates': [(1.0, 'FX CHAMBER'), (2.0, 'CM DRV')]},
        ]
    }
    selections = {('a.dxf', 0): ['FX CHAMBER']}
    rows = ref_designator.build_all_regions_summary(analysis, selections, 'a.dxf', [])
    assert rows[0]['name'] == 'FX CHAMBER'


def test_build_all_regions_summary_no_candidates_uses_no_name():
    """候補が元々無い領域は build_named_regions() と同じ番号付け規則で
    "no name" になる（自動確定扱い）。"""
    square = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    analysis = {
        'regions': [
            {'id': 0, 'frame': 0, 'polygon': square, 'area_pct': 30.0, 'name_candidates': []},
        ]
    }
    rows = ref_designator.build_all_regions_summary(analysis, {}, 'a.dxf', [])
    assert rows[0]['name'] == 'no name'


def test_build_all_regions_summary_counts_labels_regardless_of_confirmation():
    """領域内ラベル数・領域内機器符号候補数は、確定状況に関わらずポリゴン内包
    判定で数える（v3.0.0、要求「機器符号（候補）以外も抽出オプションに関係なく
    カウントする」）。未確定領域（'CM DRV' の候補）にもラベルが割り当てられる。"""
    square = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    analysis = {
        'regions': [
            {'id': 0, 'frame': 0, 'polygon': square, 'area_pct': 30.0,
             'name_candidates': [(1.0, 'CM DRV')]},   # 未確定（候補のみ）
        ]
    }
    labels = [('R10', 10, 10), ('NOTE ONE', 20, 20), ('CN3', 500, 500)]  # 500,500は矩形外
    rows = ref_designator.build_all_regions_summary(analysis, {}, 'a.dxf', labels)
    assert rows[0]['label_count'] == 2       # R10・NOTE ONE（矩形内の2件）
    assert rows[0]['ref_designator_count'] == 1   # うち機器符号候補はR10のみ


def test_build_all_regions_summary_does_not_assign_unconfirmed_name_to_labels():
    """未確定領域の候補名は、ラベル側の '領域' 列（build_labeled_rows等）には
    影響しない（意図的な非対称。build_all_regions_summary は領域一覧の集計のみ
    に使う独立した内包判定であり、build_named_regions/build_region_output の
    named には含まれないため、ラベルの '領域' 列は空のまま）。"""
    square = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    analysis = {
        'regions': [
            {'id': 0, 'frame': 0, 'polygon': square, 'area_pct': 30.0,
             'name_candidates': [(1.0, 'CM DRV')]},
        ]
    }
    labels = [('R10', 10, 10)]
    named, _ = ref_designator.build_named_regions(analysis, {}, 'a.dxf')
    rows = ref_designator.build_labeled_rows(labels, named_regions=named)
    assert rows[0]['領域'] == ''   # 未確定領域には割り当てられない


# ---------------------------------------------------------------------------
# 図面情報枠（タイトルブロック）の幾何検出（合成データ）
# ---------------------------------------------------------------------------

def _make_standard_title_block_lines(xr, y0, width=175.0, height=64.0):
    """右辺=xr・底辺=y0に接する標準的なタイトルブロック矩形の境界線を返す。"""
    left = xr - width
    top = y0 + height
    return [
        ((left, y0), (xr, y0)),   # 底辺
        ((xr, y0), (xr, top)),    # 右辺
        ((left, y0), (left, top)),  # 左辺
        ((left, top), (xr, top)),  # 上辺
    ]


def test_detect_info_frame_bbox_uses_boundary_line_when_available():
    """タイトル項目名が3件以上一致し、境界線も見つかる場合はその境界線を
    情報枠のbboxとして使う。"""
    xl, xr, y0, y1 = 0.0, 200.0, 0.0, 300.0
    block_lines = _make_standard_title_block_lines(xr, y0)
    label_points = [
        ('TITLE', 150.0, 20.0), ('DATE', 160.0, 30.0), ('NAME', 170.0, 40.0),
    ]
    bbox = ref_designator._detect_info_frame_bbox(block_lines, label_points, xl, xr, y0, y1)
    assert bbox == (xr - 175.0, xr, y0, y0 + 64.0)


def test_detect_info_frame_bbox_returns_none_when_too_few_title_terms():
    """タイトル項目名の一致が最小数未満なら情報枠なしとみなす（None）。"""
    xl, xr, y0, y1 = 0.0, 200.0, 0.0, 300.0
    label_points = [('TITLE', 150.0, 20.0), ('DATE', 160.0, 30.0)]  # 2件のみ
    bbox = ref_designator._detect_info_frame_bbox([], label_points, xl, xr, y0, y1)
    assert bbox is None


def test_detect_info_frame_bbox_exact_match_only_avoids_substring_false_positive():
    """部分一致ではなく完全一致で判定するため、回路内容の説明文にたまたま
    'CHECK' 等の語が部分文字列として含まれていても誤マッチしない
    （実データ EE6333-610-07B.dxf で確認: 'SNM-5 DI2 SET POINT (OUT CHECK)'）。"""
    xl, xr, y0, y1 = 0.0, 200.0, 0.0, 300.0
    label_points = [
        ('SNM-5 DI2 SET POINT (OUT CHECK)', 100.0, 200.0),  # 部分文字列としてCHECKを含むが除外
        ('TITLE', 150.0, 20.0), ('DATE', 160.0, 30.0),
    ]
    bbox = ref_designator._detect_info_frame_bbox([], label_points, xl, xr, y0, y1)
    assert bbox is None  # 3件に満たない（誤マッチしていれば3件になってしまう）


def test_detect_info_frame_bbox_picks_bottom_cluster_not_unrelated_top_labels():
    """底辺(y0)から離れた位置にある同名ラベル群（「流用元図番」の履歴テーブル等）は
    除外し、底辺に接する最初の連続クラスタのみを情報枠の判定に使う
    （実データ EE6868-405C.dxf で確認: y=266付近に無関係な TITLE/DWG No. の
    繰り返しがあり、これを含めるとタイトルブロックのbboxが誤って
    図面全体近くまで拡大してしまっていた）。"""
    xl, xr, y0, y1 = 0.0, 200.0, 0.0, 300.0
    block_lines = _make_standard_title_block_lines(xr, y0)
    label_points = [
        # 底辺に接するタイトルブロック本体
        ('TITLE', 150.0, 20.0), ('DATE', 160.0, 30.0), ('NAME', 170.0, 40.0),
        # 遠く離れた無関係なクラスタ（y=266、y_gap=20を大きく超える）
        ('TITLE', 50.0, 266.0), ('DATE', 60.0, 266.0), ('NAME', 70.0, 266.0),
    ]
    bbox = ref_designator._detect_info_frame_bbox(block_lines, label_points, xl, xr, y0, y1)
    assert bbox == (xr - 175.0, xr, y0, y0 + 64.0)


def test_detect_info_frame_bbox_picks_rightmost_cluster_when_side_by_side():
    """タイトルブロックが横並びに複数存在する図面（旧版タイトルブロックとの
    比較表示等）では、右辺xrに近い側のクラスタのみを採用する
    （実データ EE6868-516-01A.dxf で確認: 「流用元図番」を含む旧版
    タイトルブロックが左に、現行タイトルブロックが右に並んでいた）。"""
    xl, xr, y0, y1 = 0.0, 800.0, 0.0, 500.0
    block_lines = _make_standard_title_block_lines(xr, y0)
    label_points = [
        # 右側（本来のタイトルブロック、xrに接する）
        ('TITLE', 750.0, 20.0), ('DATE', 760.0, 30.0), ('NAME', 770.0, 40.0),
        # 左側（横並びの別タイトルブロック、y範囲は同じだがxが離れている）
        ('TITLE', 400.0, 20.0), ('DATE', 410.0, 30.0), ('NAME', 420.0, 40.0),
    ]
    bbox = ref_designator._detect_info_frame_bbox(block_lines, label_points, xl, xr, y0, y1)
    assert bbox == (xr - 175.0, xr, y0, y0 + 64.0)


def test_detect_info_frame_bbox_falls_back_to_label_bbox_when_no_boundary_line():
    """境界線が見つからない場合はラベルbbox+マージンでフォールバックする。"""
    xl, xr, y0, y1 = 0.0, 200.0, 0.0, 300.0
    label_points = [
        ('TITLE', 150.0, 20.0), ('DATE', 160.0, 30.0), ('NAME', 170.0, 40.0),
    ]
    bbox = ref_designator._detect_info_frame_bbox([], label_points, xl, xr, y0, y1)
    assert bbox is not None
    left, right, bottom, top = bbox
    assert right == xr
    assert bottom == y0
    assert left <= 150.0  # ラベルbbox左端(150)-マージン以下
    assert top >= 40.0    # ラベルbbox上端(40)+マージン以上


# ---------------------------------------------------------------------------
# 図面枠検出・図面情報枠の幾何検出（実 DXF）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sample_name', [
    'EE6868-500-01C.dxf',   # 13図面（複数図面ファイル）
    'EE6097-039-06C.dxf',   # 3図面
])
def test_titleblock_content_excluded_from_real_sample(sample_name):
    """図面情報枠内のラベル（TITLE/DATE/REVISION/人名等）が機器符号抽出結果に
    一切現れないこと（幾何検出による情報枠除外の検証）。"""
    path = _find_sample(sample_name)
    if not path:
        pytest.skip(f'sample DXF not found: {sample_name}')

    data = ref_designator.extract_ref_designator_data(
        path, frame_lineweight=100, original_filename=sample_name)
    assert data['warning'] is None
    all_texts = {t for t, _x, _y in data['labels']}

    titleblock_leakage = {'TITLE', 'DATE', 'REVISION', 'APPRV', 'CHECK', 'DESIG',
                           'DRAW', 'MARK', 'REMARKS', 'SCALE', 'NAME',
                           'KURIHARA', 'MORIOKA', 'KANAI'} & all_texts
    assert not titleblock_leakage, f'titleblock content leaked: {titleblock_leakage}'
    assert data['total_in_frame'] > 0


@pytest.mark.parametrize('sample_name', [
    'EE6868-500-01C.dxf',
    'EE6097-039-06C.dxf',
])
def test_is_ref_designator_label_flags_are_consistent_with_classify_labels(sample_name):
    """v3.0.0で `extract_ref_designator_data()['labels']` は枠内・情報枠外の
    全ラベルを保持するようになった（旧仕様では3候補パターンに一致するものだけに
    絞り込んでいた）。機器符号候補かどうかの判定自体は `is_ref_designator_label()`
    に一本化されており、`classify_labels()` で絞り込んだ結果と完全に一致する
    ことを実データで検証する。"""
    path = _find_sample(sample_name)
    if not path:
        pytest.skip(f'sample DXF not found: {sample_name}')

    data = ref_designator.extract_ref_designator_data(
        path, frame_lineweight=100, original_filename=sample_name)
    flagged = {t for t, _x, _y in data['labels'] if ref_designator.is_ref_designator_label(t)}
    classified = {t for t, _x, _y in ref_designator.classify_labels(data['labels'])}
    assert flagged == classified


# ---------------------------------------------------------------------------
# invisible エンティティの除外（2026-09-11、ユーザー報告）
# ---------------------------------------------------------------------------

def _build_doc_with_invisible_entities():
    """合成DXF: 図面枠(0,0)-(200,100)の中に、可視ラベル'R10'と、
    非表示(invisible=1)のラベル'Takahashi'・非表示の図面枠風LINE4本を配置する。"""
    doc = ezdxf.new()
    msp = doc.modelspace()
    for p1, p2 in [((0, 0), (200, 0)), ((200, 0), (200, 100)),
                   ((200, 100), (0, 100)), ((0, 100), (0, 0))]:
        msp.add_line(p1, p2, dxfattribs={'lineweight': 100, 'color': 7})
    msp.add_text('R10', dxfattribs={'insert': (10, 10)})
    invisible_text = msp.add_text('Takahashi', dxfattribs={'insert': (50, 50)})
    invisible_text.dxf.invisible = 1
    # 非表示の「図面枠風」LINE4本（可視の主枠とは別の場所、invisible=1）。
    # 実データ〈EE6892-612-01B.dxf〉では旧版タイトルブロックの罫線も
    # invisible=1 のまま残っていたことを踏まえた組み合わせ確認。
    for p1, p2 in [((300, 0), (500, 0)), ((500, 0), (500, 100)),
                   ((500, 100), (300, 100)), ((300, 100), (300, 0))]:
        line = msp.add_line(p1, p2, dxfattribs={'lineweight': 100, 'color': 7})
        line.dxf.invisible = 1
    return doc


def _save_and_collect(doc):
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as f:
        path = f.name
    doc.saveas(path)
    try:
        return ref_designator.collect_in_frame_labels(path, frame_lineweight=100)
    finally:
        os.remove(path)


def test_invisible_label_excluded_from_collected_labels():
    """invisible=1 のTEXT/MTEXTエンティティは、DXFファイル中に座標・テキストが
    実在していても、機器符号抽出結果に一切現れない（2026-09-11、ユーザー報告:
    `EE6892-612-01B.dxf` の設計者名'Takahashi'が、図面上には表示されない
    〈invisible=1〉にもかかわらず抽出結果に含まれていた不具合の修正）。"""
    result = _save_and_collect(_build_doc_with_invisible_entities())
    assert result['error'] is None
    texts = {t for (t, _x, _y) in result['labels']}
    assert 'R10' in texts
    assert 'Takahashi' not in texts


def test_invisible_line_does_not_form_extra_frame():
    """invisible=1 のLINE（図面枠と同じlineweight/colorでも）は図面枠検出の
    対象にしない。可視の主枠1つだけが検出されることを固定する。"""
    result = _save_and_collect(_build_doc_with_invisible_entities())
    assert result['error'] is None
    assert len(result['frames']) == 1
    assert result['frames'][0] == (0.0, 200.0, 0.0, 100.0)


@pytest.mark.parametrize('sample_name', ['EE6892-612-01B.dxf'])
def test_invisible_titleblock_content_excluded_from_real_sample(sample_name):
    """実データ〈EE6892-612-01B.dxf〉: 同一図面枠内に新旧2つのタイトルブロックが
    横並びで存在し、旧タイトルブロック一式（設計者名'Takahashi'を含む、
    `APPRV`/`CHECK`/`DESIG`/`DRAW`/`TITLE`等のラベルとその罫線）は全て
    invisible=1 で図面上には表示されない。これらが抽出結果に一切現れない
    ことを実データで確認する（`_detect_info_frame_bbox()`による幾何除外とは
    独立した、invisibleフラグによる除外——旧タイトルブロックは現行のものより
    左に離れた位置にあり、幾何検出の対象クラスタにそもそも含まれないため、
    この確認はinvisible除外が正しく効いていることの直接的な検証になる）。"""
    path = _find_sample(sample_name)
    if not path:
        pytest.skip(f'sample DXF not found: {sample_name}')

    data = ref_designator.extract_ref_designator_data(
        path, frame_lineweight=100, original_filename=sample_name)
    assert data['warning'] is None
    all_texts = {t for t, _x, _y in data['labels']}

    old_titleblock_leakage = {
        'Takahashi', 'APPRV', 'CHECK', 'DESIG', 'DRAW', '流用元図番',
    } & all_texts
    assert not old_titleblock_leakage, (
        f'invisible旧タイトルブロックの内容が漏れている: {old_titleblock_leakage}')


def test_invisible_excluded_from_region_geometry():
    """invisible=1 のエンティティは region_detector._collect_region_geometry()
    の収集対象にもならない（2026-09-11、ref_designator.py以外の収集経路への
    横展開）。直接配置の境界LINE・TEXT・INSERT自身（CIRCLE内包ブロックの
    接続点カウント）・INSERT展開後のTEXTの4パターンを確認する。"""
    doc = ezdxf.new()
    msp = doc.modelspace()
    cfg = DEFAULT_REGION_CONFIG

    # 可視の領域境界LINE（lineweight=25, color=2）
    msp.add_line((0, 0), (10, 0), dxfattribs={'lineweight': 25, 'color': 2})
    # 非表示の領域境界LINE（同じ識別キーだが invisible=1）
    invisible_line = msp.add_line((0, 20), (10, 20), dxfattribs={'lineweight': 25, 'color': 2})
    invisible_line.dxf.invisible = 1

    # 可視ラベルと非表示ラベル
    msp.add_text('VISIBLE', dxfattribs={'insert': (0, 5)})
    invisible_text = msp.add_text('HIDDEN', dxfattribs={'insert': (0, 25)})
    invisible_text.dxf.invisible = 1

    # CIRCLE を含むブロック（接続点候補）
    circle_block = doc.blocks.new(name='CONN_PT')
    circle_block.add_circle((0, 0), radius=1)

    # 可視INSERT（接続点として数えられるべき）
    msp.add_blockref('CONN_PT', (50, 50))
    # 非表示INSERT（中身ごと除外され、接続点として数えられてはならない）
    invisible_insert = msp.add_blockref('CONN_PT', (60, 60))
    invisible_insert.dxf.invisible = 1

    # テキストを含むブロック（INSERT展開後のTEXT収集経路の確認用）
    text_block = doc.blocks.new(name='TEXT_BLK')
    text_block.add_text('FROM_INSERT', dxfattribs={'insert': (0, 0)})
    msp.add_blockref('TEXT_BLK', (100, 100))

    frame_lines, region_lines, region_lines_lp, label_entities, connection_points = (
        _collect_region_geometry(doc, cfg))

    label_texts = {e.dxf.text for e in label_entities if e.dxftype() == 'TEXT'}
    assert 'VISIBLE' in label_texts
    assert 'HIDDEN' not in label_texts
    assert 'FROM_INSERT' in label_texts
    assert len(region_lines) == 1  # 可視LINEのみ
    assert len(connection_points) == 1  # 可視INSERTのみ


def test_invisible_excluded_from_terminal_geometry():
    """invisible=1 のエンティティは terminal_detector._collect_geometry()
    の収集対象にもならない（2026-09-11、横展開）。TEXT・端子台矩形辺LINE・
    橋渡しCIRCLEの3種、および直接配置・INSERT自身・INSERT展開後のいずれの
    経路でも除外されることを確認する。"""
    from model.terminal_detector import TB_LINE_WEIGHT, TB_LINE_COLOR, TB_CIRCLE_WEIGHT, TB_CIRCLE_COLOR

    doc = ezdxf.new()
    msp = doc.modelspace()

    # 可視の端子台矩形辺LINE
    msp.add_line((0, 0), (10, 0), dxfattribs={'lineweight': TB_LINE_WEIGHT, 'color': TB_LINE_COLOR})
    # 非表示の端子台矩形辺LINE
    invisible_line = msp.add_line(
        (0, 20), (10, 20), dxfattribs={'lineweight': TB_LINE_WEIGHT, 'color': TB_LINE_COLOR})
    invisible_line.dxf.invisible = 1

    # 可視の橋渡しCIRCLE
    msp.add_circle((5, 5), radius=1, dxfattribs={'lineweight': TB_CIRCLE_WEIGHT, 'color': TB_CIRCLE_COLOR})
    # 非表示の橋渡しCIRCLE
    invisible_circle = msp.add_circle(
        (5, 25), radius=1, dxfattribs={'lineweight': TB_CIRCLE_WEIGHT, 'color': TB_CIRCLE_COLOR})
    invisible_circle.dxf.invisible = 1

    # 可視テキストと非表示テキスト
    msp.add_text('VISIBLE', dxfattribs={'insert': (0, 5)})
    invisible_text = msp.add_text('HIDDEN', dxfattribs={'insert': (0, 25)})
    invisible_text.dxf.invisible = 1

    # 非表示INSERT（中身ごと除外されるべき）
    text_block = doc.blocks.new(name='TB_TEXT_BLK')
    text_block.add_text('FROM_INSERT_HIDDEN', dxfattribs={'insert': (0, 0)})
    invisible_insert = msp.add_blockref('TB_TEXT_BLK', (100, 100))
    invisible_insert.dxf.invisible = 1

    texts, lines, circles = _collect_terminal_geometry(doc)

    text_values = {t for t, _x, _y, _rot in texts}
    assert 'VISIBLE' in text_values
    assert 'HIDDEN' not in text_values
    assert 'FROM_INSERT_HIDDEN' not in text_values
    assert len(lines) == 1  # 可視LINEのみ
    assert len(circles) == 1  # 可視CIRCLEのみ


def test_invisible_excluded_from_extract_labels_direct_call():
    """invisible=1 のエンティティは extract_labels()（機器符号抽出とは独立した
    ラベル抽出APIの本体）でも収集されない（2026-09-11、横展開）。MODEL_SPACE
    直接配置・INSERT展開の2経路を、既存のref_designator経由テストとは別に
    extract_labels()を直接呼んで確認する。"""
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_text('VISIBLE_DIRECT', dxfattribs={'insert': (0, 0)})
    invisible_text = msp.add_text('HIDDEN_DIRECT', dxfattribs={'insert': (10, 10)})
    invisible_text.dxf.invisible = 1

    text_block = doc.blocks.new(name='LBL_BLK')
    text_block.add_text('VISIBLE_FROM_INSERT', dxfattribs={'insert': (0, 0)})
    msp.add_blockref('LBL_BLK', (50, 50))

    hidden_block = doc.blocks.new(name='HIDDEN_LBL_BLK')
    hidden_block.add_text('HIDDEN_FROM_INSERT', dxfattribs={'insert': (0, 0)})
    hidden_insert = msp.add_blockref('HIDDEN_LBL_BLK', (60, 60))
    hidden_insert.dxf.invisible = 1

    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as f:
        path = f.name
    doc.saveas(path)
    try:
        labels, info = extract_labels(path)
    finally:
        os.remove(path)

    assert 'VISIBLE_DIRECT' in labels
    assert 'HIDDEN_DIRECT' not in labels
    assert 'VISIBLE_FROM_INSERT' in labels
    assert 'HIDDEN_FROM_INSERT' not in labels


def _build_doc_with_hidden_layer_entities():
    """合成DXF: 図面枠(0,0)-(200,100)の中に、可視ラベル'R10'（レイヤー'0'）と、
    off+frozenの専用レイヤーに置かれたラベル'OldDesigner'を配置する
    （2026-09-16、ユーザー報告: entity自身のinvisible属性ではなく、旧
    タイトルブロックを専用レイヤーごとoff/frozenにして非表示にする運用が
    あり、実データ`EE3273-039-90B.dxf`で誤って抽出されていた）。"""
    doc = ezdxf.new()
    msp = doc.modelspace()
    for p1, p2 in [((0, 0), (200, 0)), ((200, 0), (200, 100)),
                   ((200, 100), (0, 100)), ((0, 100), (0, 0))]:
        msp.add_line(p1, p2, dxfattribs={'lineweight': 100, 'color': 7})
    msp.add_text('R10', dxfattribs={'insert': (10, 10)})

    hidden_layer_name = 'OLD_TITLEBLOCK_LAYER'
    doc.layers.add(hidden_layer_name)
    hidden_layer = doc.layers.get(hidden_layer_name)
    hidden_layer.off()
    hidden_layer.freeze()
    msp.add_text('OldDesigner', dxfattribs={'insert': (50, 50), 'layer': hidden_layer_name})
    return doc


def test_layer_off_frozen_label_excluded_from_collected_labels():
    """entity自身のinvisible属性は立っていなくても、置かれたレイヤーが
    off+frozenであるTEXTは機器符号抽出結果に一切現れない（2026-09-16、
    is_invisible()へのレイヤー可視性チェック追加の直接の検証）。"""
    result = _save_and_collect(_build_doc_with_hidden_layer_entities())
    assert result['error'] is None
    texts = {t for (t, _x, _y) in result['labels']}
    assert 'R10' in texts
    assert 'OldDesigner' not in texts


@pytest.mark.parametrize('off,frozen', [(True, False), (False, True), (True, True)])
def test_layer_off_or_frozen_alone_excludes_entity(off, frozen):
    """off単独・frozen単独・両方のいずれでも除外される（is_invisible()の
    `layer.is_off() or layer.is_frozen()`のOR条件を分岐ごとに固定する）。"""
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_text('VISIBLE', dxfattribs={'insert': (0, 0)})
    layer_name = 'HIDDEN_LAYER'
    doc.layers.add(layer_name)
    layer = doc.layers.get(layer_name)
    if off:
        layer.off()
    if frozen:
        layer.freeze()
    msp.add_text('HIDDEN', dxfattribs={'insert': (10, 10), 'layer': layer_name})

    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as f:
        path = f.name
    doc.saveas(path)
    try:
        labels, info = extract_labels(path)
    finally:
        os.remove(path)
    assert 'VISIBLE' in labels
    assert 'HIDDEN' not in labels


def test_layer_off_frozen_excluded_from_region_geometry():
    """off+frozenレイヤーのエンティティは region_detector._collect_region_geometry()
    の収集対象にもならない（2026-09-16、レイヤー可視性チェックの横展開）。
    直接配置の境界LINE・TEXT・INSERT自身（CIRCLE内包ブロックの接続点カウント）・
    INSERT展開後のTEXTの4パターンを確認する（test_invisible_excluded_from_
    region_geometryのレイヤー版）。"""
    doc = ezdxf.new()
    msp = doc.modelspace()
    cfg = DEFAULT_REGION_CONFIG

    hidden_layer_name = 'HIDDEN_LAYER'
    doc.layers.add(hidden_layer_name)
    hidden_layer = doc.layers.get(hidden_layer_name)
    hidden_layer.off()
    hidden_layer.freeze()

    # 可視の領域境界LINE
    msp.add_line((0, 0), (10, 0), dxfattribs={'lineweight': 25, 'color': 2})
    # 非表示レイヤー上の領域境界LINE（同じ識別キーだがレイヤーがoff+frozen）
    msp.add_line((0, 20), (10, 20), dxfattribs={'lineweight': 25, 'color': 2, 'layer': hidden_layer_name})

    # 可視ラベルと非表示レイヤー上のラベル
    msp.add_text('VISIBLE', dxfattribs={'insert': (0, 5)})
    msp.add_text('HIDDEN', dxfattribs={'insert': (0, 25), 'layer': hidden_layer_name})

    # CIRCLE を含むブロック（接続点候補）
    circle_block = doc.blocks.new(name='CONN_PT')
    circle_block.add_circle((0, 0), radius=1)

    # 可視INSERT（接続点として数えられるべき）
    msp.add_blockref('CONN_PT', (50, 50))
    # 非表示レイヤー上のINSERT（中身ごと除外され、接続点として数えられてはならない）
    msp.add_blockref('CONN_PT', (60, 60), dxfattribs={'layer': hidden_layer_name})

    # テキストを含むブロック（INSERT展開後のTEXT収集経路の確認用）
    text_block = doc.blocks.new(name='TEXT_BLK')
    text_block.add_text('FROM_INSERT', dxfattribs={'insert': (0, 0)})
    msp.add_blockref('TEXT_BLK', (100, 100))

    frame_lines, region_lines, region_lines_lp, label_entities, connection_points = (
        _collect_region_geometry(doc, cfg))

    label_texts = {e.dxf.text for e in label_entities if e.dxftype() == 'TEXT'}
    assert 'VISIBLE' in label_texts
    assert 'HIDDEN' not in label_texts
    assert 'FROM_INSERT' in label_texts
    assert len(region_lines) == 1  # 可視LINEのみ
    assert len(connection_points) == 1  # 可視INSERTのみ


def test_layer_off_frozen_excluded_from_terminal_geometry():
    """off+frozenレイヤーのエンティティは terminal_detector._collect_geometry()
    の収集対象にもならない（2026-09-16、横展開）。TEXT・端子台矩形辺LINE・
    橋渡しCIRCLEの3種、および直接配置・INSERT展開後のいずれの経路でも
    除外されることを確認する（test_invisible_excluded_from_terminal_geometryの
    レイヤー版）。"""
    from model.terminal_detector import TB_LINE_WEIGHT, TB_LINE_COLOR, TB_CIRCLE_WEIGHT, TB_CIRCLE_COLOR

    doc = ezdxf.new()
    msp = doc.modelspace()

    hidden_layer_name = 'HIDDEN_LAYER'
    doc.layers.add(hidden_layer_name)
    hidden_layer = doc.layers.get(hidden_layer_name)
    hidden_layer.off()
    hidden_layer.freeze()

    # 可視の端子台矩形辺LINE
    msp.add_line((0, 0), (10, 0), dxfattribs={'lineweight': TB_LINE_WEIGHT, 'color': TB_LINE_COLOR})
    # 非表示レイヤー上の端子台矩形辺LINE
    msp.add_line((0, 20), (10, 20), dxfattribs={
        'lineweight': TB_LINE_WEIGHT, 'color': TB_LINE_COLOR, 'layer': hidden_layer_name})

    # 可視の橋渡しCIRCLE
    msp.add_circle((5, 5), radius=1, dxfattribs={'lineweight': TB_CIRCLE_WEIGHT, 'color': TB_CIRCLE_COLOR})
    # 非表示レイヤー上の橋渡しCIRCLE
    msp.add_circle((5, 25), radius=1, dxfattribs={
        'lineweight': TB_CIRCLE_WEIGHT, 'color': TB_CIRCLE_COLOR, 'layer': hidden_layer_name})

    # 可視テキストと非表示レイヤー上のテキスト
    msp.add_text('VISIBLE', dxfattribs={'insert': (0, 5)})
    msp.add_text('HIDDEN', dxfattribs={'insert': (0, 25), 'layer': hidden_layer_name})

    # 非表示レイヤー上のINSERT（中身ごと除外されるべき）
    text_block = doc.blocks.new(name='TB_TEXT_BLK')
    text_block.add_text('FROM_INSERT_HIDDEN', dxfattribs={'insert': (0, 0)})
    msp.add_blockref('TB_TEXT_BLK', (100, 100), dxfattribs={'layer': hidden_layer_name})

    texts, lines, circles = _collect_terminal_geometry(doc)

    text_values = {t for t, _x, _y, _rot in texts}
    assert 'VISIBLE' in text_values
    assert 'HIDDEN' not in text_values
    assert 'FROM_INSERT_HIDDEN' not in text_values
    assert len(lines) == 1  # 可視LINEのみ
    assert len(circles) == 1  # 可視CIRCLEのみ


def test_layer_off_frozen_excluded_from_extract_labels_direct_call():
    """off+frozenレイヤーのエンティティは extract_labels()（機器符号抽出とは
    独立したラベル抽出APIの本体）でも収集されない（2026-09-16、横展開）。
    MODEL_SPACE直接配置・INSERT展開の2経路を確認する
    （test_invisible_excluded_from_extract_labels_direct_callのレイヤー版）。"""
    doc = ezdxf.new()
    msp = doc.modelspace()

    hidden_layer_name = 'HIDDEN_LAYER'
    doc.layers.add(hidden_layer_name)
    hidden_layer = doc.layers.get(hidden_layer_name)
    hidden_layer.off()
    hidden_layer.freeze()

    msp.add_text('VISIBLE_DIRECT', dxfattribs={'insert': (0, 0)})
    msp.add_text('HIDDEN_DIRECT', dxfattribs={'insert': (10, 10), 'layer': hidden_layer_name})

    text_block = doc.blocks.new(name='LBL_BLK')
    text_block.add_text('VISIBLE_FROM_INSERT', dxfattribs={'insert': (0, 0)})
    msp.add_blockref('LBL_BLK', (50, 50))

    hidden_block = doc.blocks.new(name='HIDDEN_LBL_BLK')
    hidden_block.add_text('HIDDEN_FROM_INSERT', dxfattribs={'insert': (0, 0)})
    msp.add_blockref('HIDDEN_LBL_BLK', (60, 60), dxfattribs={'layer': hidden_layer_name})

    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as f:
        path = f.name
    doc.saveas(path)
    try:
        labels, info = extract_labels(path)
    finally:
        os.remove(path)

    assert 'VISIBLE_DIRECT' in labels
    assert 'HIDDEN_DIRECT' not in labels
    assert 'VISIBLE_FROM_INSERT' in labels
    assert 'HIDDEN_FROM_INSERT' not in labels


@pytest.mark.slow
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
    # 実データ検証（sample-dxf/全156件比較）で全13枠が175×64の標準タイトル
    # ブロックとして検出できることを確認済み（frames_without_info_frame=0）。
    assert result['frames_without_info_frame'] == 0


@pytest.mark.parametrize('sample_name', [
    'EE5322-455-01B.dxf',   # 図面枠(JZB_0001)はPaper Space("ICADSX Layout")のみ、
    'EE5322-455-07A.dxf',   # 実ジオメトリはModel Space（frameとcontentが別レイアウト）
])
def test_frame_absent_in_model_space_falls_back_without_losing_labels(sample_name):
    """図面枠を含む内容一式が Model Space ではなく Paper Space レイアウトに
    配置されている一方、実際のラベル・図形は Model Space にある図面
    （frame と content がレイアウト的に分離した構造）では、Paper Space 側の
    図面枠を Model Space のラベルに適用してはならない（Model/Paper Space は
    独立した座標系のため、座標が対応せず正しいラベルの大半が「枠外」と
    誤判定される。実際に `EE6892-455B.dxf` で `CB001` 等が出力から消える
    不具合が発生した。2026-07-14 ユーザー報告により発覚）。

    このため、Model Space に何らかのラベルがある場合は Model Space のみを
    対象にする方針とし、Model Space に図面枠が無ければ「図面枠が見つかりま
    せんでした」という警告付きで図面枠フィルタなしの全ラベル抽出にフォール
    バックする（Paper Space の図面枠を誤って適用しない代わりに、警告が出る
    のは許容する）。"""
    path = _find_sample(sample_name)
    if not path:
        pytest.skip(f'sample DXF not found: {sample_name}')
    data = ref_designator.extract_ref_designator_data(
        path, frame_lineweight=100, original_filename=sample_name)
    assert data['warning'] is not None
    # 図面枠フィルタなしのフォールバックで、Model Space の実ラベルが
    # ちゃんと収集されている（座標系不一致でほぼ全滅していない）こと。
    # 閾値は20→15に緩和（2026-09-23）: `EE5322-455-07A.dxf`（本サンプルの
    # 内容）では、このフォールバック経路
    # （`_collect_all_labels_fallback()`）に非表示エンティティの除外
    # （`is_invisible()`）が無く、entity自身のinvisible属性が立った
    # 6件（`I`×2・`J`×2・`I-I(1:1)`・`J-J(1:1)`、いずれも図面上に表示
    # されない）が出力に混入していた（25件）。除外を追加した結果19件に
    # 減っており、これは正しい修正の結果（ほぼ全滅の兆候ではない）。
    assert data['total_in_frame'] > 15


@pytest.mark.slow
def test_cb001_not_dropped_by_paper_space_frame_coordinate_mismatch():
    """ユーザー報告の具体的な再現ケース: `EE6892-455B.dxf` は図面枠が
    Paper Space（`ICADSX Layout`）のみに存在し、実際の機器符号ラベル
    （`CB001`等）は Model Space にある。Paper Space 由来の図面枠bboxを
    Model Space のラベルに誤って適用すると、座標系が対応せず`CB001`等の
    大半が「枠外」と誤判定されて出力から消える不具合が発生した
    （2026-07-14、修正前は 60件→6件に激減していた）。"""
    path = _find_sample('EE6892-455B.dxf')
    if not path:
        pytest.skip('sample DXF not found: EE6892-455B.dxf')
    data = ref_designator.extract_ref_designator_data(
        path, frame_lineweight=100, original_filename='EE6892-455B.dxf')
    all_labels = {t for t, _x, _y in data['labels']}
    assert 'CB001' in all_labels
    assert len(all_labels) > 50


def test_extract_ref_designator_rows_end_to_end():
    """通常モード用トップレベル関数が実DXFに対し例外なく行データを返す。"""
    path = _find_sample('EE6491-039-04A.dxf')
    if not path:
        pytest.skip('sample DXF not found: EE6491-039-04A.dxf')
    result = ref_designator.extract_ref_designator_rows(
        path, frame_lineweight=100, original_filename='EE6491-039-04A.dxf')
    assert result['filename'] == 'EE6491-039-04A.dxf'
    assert result['warning'] is None
    assert result['frames'] > 0
    assert len(result['rows']) > 0
    # 行データは {'機器符号候補': str|None, 'ラベル': str, '個数': int} の形（v3.0.0）
    for sample in result['rows'][:5]:
        assert set(sample.keys()) == {'機器符号候補', 'ラベル', '個数'}


@pytest.mark.slow
def test_formerly_excluded_words_now_appear_when_outside_info_frame():
    """旧仕様では除外パターンにより弾かれていた語（GND 等の回路説明語）が、
    図面情報枠の外にある限り新仕様では出力に含まれること
    （2026-07-30 ユーザー承認の仕様変更そのものの検証。除外パターン廃止の
    直接の帰結）。"""
    path = _find_sample('EE6868-500-01C.dxf')
    if not path:
        pytest.skip('sample DXF not found: EE6868-500-01C.dxf')
    data = ref_designator.extract_ref_designator_data(
        path, frame_lineweight=100, original_filename='EE6868-500-01C.dxf')
    all_labels = {t for t, _x, _y in data['labels']}
    assert 'GND' in all_labels
