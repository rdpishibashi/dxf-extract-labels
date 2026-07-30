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

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model import ref_designator  # noqa: E402

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
    （name_selections）が無い限り named に含まれない（build_region_results と同じ挙動）。
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
    ref_data_by_file = {
        'a.dxf': {
            'labels': [('R10', 0, 0), ('CN3', 1, 1)],
            'total_in_frame': 5,
            'warning': None,
            'main_drawing_number': 'EE1234-500-01A',
            'source_drawing_number': None,
            'title': 'TITLE-A',
            'subtitle': None,
        },
    }
    ref_final = ref_designator.build_ref_designator_final(ref_data_by_file, sort_value='asc')
    assert ref_final['a.dxf']['rows'] == [
        {'ラベル': 'CN3', '個数': 1}, {'ラベル': 'R10', '個数': 1}]
    assert ref_final['a.dxf']['total_in_frame'] == 5
    assert ref_final['a.dxf']['main_drawing_number'] == 'EE1234-500-01A'
    assert ref_final['a.dxf']['title'] == 'TITLE-A'


def test_build_ref_designator_region_results():
    square = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    ref_data_by_file = {
        'a.dxf': {'labels': [('R10', 10, 10), ('CN3', 200, 200)], 'total_in_frame': 4},
    }
    region_analyses = {
        'a.dxf': {
            'frames': [(0, 0, 0, 0)],
            'regions': [{'id': 0, 'frame': 0, 'polygon': square, 'area_pct': 50.0,
                         'name_candidates': [(1.0, 'RACK1')]}],
            'main_drawing_number': 'EE1234-500-01A',
        },
    }
    name_selections_by_file = {'a.dxf': {('a.dxf', 0): ['RACK1']}}
    results = ref_designator.build_ref_designator_region_results(
        ref_data_by_file, region_analyses, name_selections_by_file, sort_value='asc')
    r = results['a.dxf']
    assert r['total_in_frame'] == 4
    assert r['final_count'] == 2   # R10 + CN3
    assert r['filtered_count'] == 2   # total_in_frame(4) - final_count(2)
    by_label = {row['ラベル']: row for row in r['rows']}
    assert by_label['R10']['領域'] == 'RACK1'
    assert by_label['CN3']['領域'] == ''


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
def test_output_labels_all_match_candidate_pattern(sample_name):
    """出力される機器符号はすべて3候補パターンのいずれかに一致すること
    （判定は括弧より前の部分に対して行う）。除外パターンは廃止されたため、
    ここでは形式一致のみを検証する。"""
    path = _find_sample(sample_name)
    if not path:
        pytest.skip(f'sample DXF not found: {sample_name}')

    data = ref_designator.extract_ref_designator_data(
        path, frame_lineweight=100, original_filename=sample_name)
    for text, _x, _y in data['labels']:
        judgment = ref_designator._judgment_text(text)
        assert ref_designator.CANDIDATE_PATTERN.match(judgment), (
            f'{text!r} does not match CANDIDATE_PATTERN')


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
    assert data['total_in_frame'] > 20


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
    assert len(result['rows']) > 0
    # 行データは {'ラベル': str, '個数': int} の形
    for sample in result['rows'][:5]:
        assert set(sample.keys()) == {'ラベル', '個数'}


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
