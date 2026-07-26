"""機器符号（候補）抽出モジュール（v1.6.0）— DXF抽出パイプライン

reference_designator_candidates.xlsx（`Patterns` / `ExclusionPatterns` /
`ConfirmedPatterns` シート）を正としてパターン・除外・確定リストを実装する。
DXF-extract-labels 固有の新機能であり、他プロジェクトとの共有コピーは無い
（`model/extract_labels.py` とは独立に保つ）。

**2026-07-26 モジュール分割**: 純粋なパターン判定ロジック（PATTERN_CATEGORIES・
EXCLUSION_*・CONFIRMED_PATTERN_CATEGORIES・classify_judgment_detailed 等、
ezdxf/region_detector.py への依存を持たない部分）は `ref_designator_patterns.py`
へ分離した。本ファイルは DXF ファイルの読み込み・図面枠検出・ラベル収集・
Excel出力用の行データ構築という「パイプライン」部分のみを担当する
（`region_detector.py` の `detect_drawing_frames`/`assign_region_labels` に
依存する）。分類ロジックの利用者（`app.py`・`tools/reference_designator_analyzer.py`
・テスト）は引き続き `ref_designator.classify_judgment_detailed` 等の形で
アクセスできるよう、本ファイルが `ref_designator_patterns.py` から再エクスポート
している（分割前との後方互換のため）。

処理の流れ:
  1. 図面枠（lineweight=frame_lineweight かつ color=7 の LINE 4本で構成）を検出
  2. 図面枠を構成するブロック（フォーマットブロック。図面情報欄・枠外位置記号
     A-F/1-8 を含む）由来の TEXT/MTEXT を丸ごと除外
  3. 図面枠内に残ったラベルを NFKC 正規化し、括弧より前の部分でパターン判定
  4. 3パターンのいずれかに一致し、かつ除外パターンに該当しないものを
     「機器符号（候補）」とする（reference_designator_candidates.xlsx の
     RemainingUnclassified シートと同じ母集団）。除外パターンに該当したもの
     （GND・TITLE・N24 等）・3パターンいずれにも一致しない文字列（`(2/5)` 等の
     記号・注記）はどちらも候補に含めない。
  5. 機器符号（候補）のうち確定パターン（`CONFIRMED_PATTERN_CATEGORIES`、
     2026-07-10）に一致するものは「確定」として自動採用される
     （「未確定ラベル」UI には表示されない）。
  6. 確定パターンに一致しなかった残りが「未確定ラベル」UI で全件レビュー対象と
     なり、ユーザーがチェックしたものだけが最終的な機器符号として出力される
     （初期状態は全て未選択）。
"""
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import ezdxf

from .common_utils import normalize_width, select_layout_result
from .extract_labels import extract_text_from_entity, _block_has_text_content
from .region_detector import detect_drawing_frames, assign_region_labels
from .ref_designator_patterns import (  # noqa: F401 (再エクスポート、後方互換用)
    PATTERNS_VERSION,
    PATTERN_CATEGORIES,
    CANDIDATE_PATTERN,
    matched_pattern_name,
    EXCLUSION_EXACT_CATEGORIES,
    EXCLUSION_REGEX_CATEGORIES,
    _compile_shorthand_pattern,
    ADDED_EXCLUSION_PATTERNS,
    ADDED_DESIGNATOR_PATTERNS,
    matched_added_designator_category,
    matched_added_exclusion_category,
    normalize_label,
    _judgment_text,
    classify_judgment_detailed,
    _classify_judgment,
    is_ref_designator_candidate,
    split_candidates,
    summarize_labels,
    sibling_key,
    propagate_selection_all_files,
    CONFIRMED_PATTERN_CATEGORIES,
    matched_confirmed_category,
    is_confirmed_designator,
)


# ============================================================
# 4. 図面枠検出・フォーマットブロック（図面情報欄）の構造的除外
# ============================================================
#
# 図面枠線は lineweight=frame_lineweight かつ color=7 の LINE 4本で構成される
# （region_detector.py の DEFAULT_REGION_CONFIG と同じ判定。lineweight 単独では
# 無関係な線分を誤って拾い図面枠検出が壊れることを実データで確認済みのため、
# color=7 の併用を維持する。2026-07-10 ユーザー確認）。
#
# 図面情報欄（右下のタイトルブロック）・図面枠外の位置記号（A-F, 1-8 等）は、
# 図面枠線を直接の子として持つ「フォーマットブロック」（実データでは JZB_*）の
# INSERT 由来であることを確認済み（サンプル18件で検証）。人名は増減するため、
# 個別の人名リストではなくこの構造的除外で対応する。

_FRAME_COLOR = 7


def _format_block_names(doc, frame_lineweight: int, frame_color: int = _FRAME_COLOR) -> set:
    """図面枠線を直接の子として持つブロック名の集合を返す（フォーマットブロック）。

    ネストされた INSERT の中までは辿らない（frame線はブロック定義の直接の子に
    置かれる想定。実データで確認済み）。
    """
    names = set()
    for blk in doc.blocks:
        for x in blk:
            if (x.dxftype() == 'LINE'
                    and getattr(x.dxf, 'lineweight', None) == frame_lineweight
                    and getattr(x.dxf, 'color', None) == frame_color):
                names.add(blk.name)
                break
    return names


def _collect_frame_and_labels(doc, frame_lineweight: int, frame_color: int = _FRAME_COLOR):
    """図面枠線と、フォーマットブロック由来を除いたラベルエンティティを収集する。

    レイアウトの選び方は `common_utils.select_layout_result()` を参照
    （Model Space に何かあれば常にそちらを使い、完全に空の場合のみ他
    レイアウトを順に試す。レイアウトをまたいで混在させない）。

    戻り値: (frame_lines, label_entities)
      frame_lines: [(start, end), ...]（Vec3のまま。detect_drawing_frames に渡す）
      label_entities: TEXT/MTEXT エンティティのリスト（フォーマットブロック由来を除く）
    """
    fmt_blocks = _format_block_names(doc, frame_lineweight, frame_color)
    text_block_cache = {}

    def is_frame_line(e):
        return (getattr(e.dxf, 'lineweight', None) == frame_lineweight
                and getattr(e.dxf, 'color', None) == frame_color)

    def collect_from_layout(layout):
        frame_lines = []
        label_entities = []
        for e in layout:
            t = e.dxftype()
            if t == 'LINE':
                if is_frame_line(e):
                    frame_lines.append((e.dxf.start, e.dxf.end))
            elif t in ('TEXT', 'MTEXT'):
                label_entities.append(e)
            elif t == 'INSERT':
                name = e.dxf.name
                if name in fmt_blocks:
                    # フォーマットブロック: 図面枠線のみ収集し、テキスト（図面情報欄・
                    # 枠外位置記号）は丸ごと除外する
                    try:
                        for v in e.virtual_entities():
                            if v.dxftype() == 'LINE' and is_frame_line(v):
                                frame_lines.append((v.dxf.start, v.dxf.end))
                    except Exception:
                        pass
                else:
                    if not _block_has_text_content(doc, name, text_block_cache):
                        continue
                    try:
                        for v in e.virtual_entities():
                            if v.dxftype() in ('TEXT', 'MTEXT'):
                                label_entities.append(v)
                    except Exception:
                        pass
        return frame_lines, label_entities

    return select_layout_result(doc, collect_from_layout, is_empty=lambda r: not any(r))


def collect_in_frame_labels(
    dxf_file: str,
    frame_lineweight: int = 100,
    frame_color: int = _FRAME_COLOR,
    snap: float = 2.0,
) -> Dict:
    """図面枠内・フォーマットブロック外のラベルを収集する。

    戻り値 dict:
      frames: [(xl,xr,y0,y1), ...]
      labels: [(text, x, y), ...]（重複除去済み、正規化前の原文）
      error: str | None
    """
    result = {'frames': [], 'labels': [], 'error': None}
    try:
        doc = ezdxf.readfile(dxf_file)
    except Exception as e:
        result['error'] = f'DXFファイルの読み込みに失敗しました: {e}'
        return result

    frame_lines, label_entities = _collect_frame_and_labels(doc, frame_lineweight, frame_color)
    frames = detect_drawing_frames(frame_lines, snap)
    result['frames'] = frames

    if not frames:
        result['error'] = (
            f'図面枠（太さ {frame_lineweight} の線で囲まれた枠）が見つかりませんでした。'
        )
        return result

    seen = set()
    labels = []
    for it in label_entities:
        _, clean_text, (x, y) = extract_text_from_entity(it)
        if not clean_text:
            continue
        in_frame = any(
            xl - 1 <= x <= xr + 1 and y0 - 1 <= y <= y1 + 1
            for (xl, xr, y0, y1) in frames
        )
        if not in_frame:
            continue
        key = (clean_text, round(x, 1), round(y, 1))
        if key in seen:
            continue
        seen.add(key)
        labels.append((clean_text, x, y))

    result['labels'] = labels
    return result


def _collect_all_labels_fallback(dxf_file: str) -> List[Tuple[str, float, float]]:
    """図面枠が検出できない場合のフォールバック: 図面枠フィルタなしで
    ファイル全体のラベルを収集する。

    レイアウトの選び方は `common_utils.select_layout_result()` を参照
    （Model Space に何かあれば常にそちらを使い、完全に空の場合のみ他
    レイアウトを順に試す）。"""
    try:
        doc = ezdxf.readfile(dxf_file)
    except Exception:
        return []
    text_block_cache = {}

    def collect_from_layout(layout):
        out = []
        for e in layout:
            t = e.dxftype()
            if t in ('TEXT', 'MTEXT'):
                _, clean_text, (x, y) = extract_text_from_entity(e)
                if clean_text:
                    out.append((clean_text, x, y))
            elif t == 'INSERT':
                if not _block_has_text_content(doc, e.dxf.name, text_block_cache):
                    continue
                try:
                    for v in e.virtual_entities():
                        if v.dxftype() in ('TEXT', 'MTEXT'):
                            _, clean_text, (x, y) = extract_text_from_entity(v)
                            if clean_text:
                                out.append((clean_text, x, y))
                except Exception:
                    pass
        return out

    return select_layout_result(doc, collect_from_layout, is_empty=lambda r: not r)


# ============================================================
# 5. ラベル正規化・機器符号候補/確定/未確定ラベルへの分類（座標付き）
# ============================================================

def normalize_labels(
    labels: List[Tuple[str, float, float]],
) -> List[Tuple[str, float, float]]:
    """(text,x,y) リストの text を NFKC 正規化+前後空白除去する（座標は保持）。
    正規化後に空文字列になったものは除く。"""
    out = []
    for (t, x, y) in labels:
        nt = normalize_label(t)
        if nt:
            out.append((nt, x, y))
    return out


def classify_labels(
    labels: List[Tuple[str, float, float]],
) -> List[Tuple[str, float, float]]:
    """正規化済み (text,x,y) リストから機器符号（候補）だけを抽出して返す。

    候補は Patterns（3パターン）に一致し、かつ除外パターンに該当しないもの
    （reference_designator_candidates.xlsx の RemainingUnclassified シートと
    同じ母集団）。除外パターン該当（GND・TITLE・N24 等）・3パターン非一致
    （`(2/5)` 等の記号・注記）はいずれも結果に含めない（2026-07-10、実データで
    RemainingUnclassified の中身を再確認して確定）。この候補集合はさらに
    `split_confirmed()` で「確定（自動採用）」/「未確定（要レビュー）」に分かれる。
    """
    return [item for item in labels
            if _classify_judgment(_judgment_text(item[0]), item[0]) == 'candidate']


def classify_excluded_labels(
    labels: List[Tuple[str, float, float]],
) -> List[Tuple[str, float, float]]:
    """正規化済み (text,x,y) リストから、3パターンには一致したが除外パターンに
    該当し Reference Designator ではないと判断されたラベルだけを抽出する
    （`classify_labels()` の 'candidate' 判定の対となる 'excluded' 判定。
    「除外符号」シート用、2026-07-26 追加）。3パターンいずれにも一致しない
    文字列（`(2/5)` 等の記号・注記、'no_match'）は対象外。
    """
    return [item for item in labels
            if _classify_judgment(_judgment_text(item[0]), item[0]) == 'excluded']


def split_confirmed(
    labels: List[Tuple[str, float, float]],
) -> Tuple[List[Tuple[str, float, float]], List[Tuple[str, float, float]]]:
    """機器符号（候補）の (text,x,y) リストを (確定, 未確定) に分ける。

    確定は確定パターン（`CONFIRMED_PATTERN_CATEGORIES`）に一致するもの。
    「未確定ラベル」UI でのレビューを経ずに最終出力へ自動採用してよい
    （2026-07-10 ユーザー指定）。それ以外が「未確定ラベル」UI でのレビュー対象。
    """
    confirmed = []
    review = []
    for item in labels:
        if matched_confirmed_category(item[0]):
            confirmed.append(item)
        else:
            review.append(item)
    return confirmed, review


# ============================================================
# 6. 通常モード（領域なし）用トップレベル関数
# ============================================================

def extract_ref_designator_data(
    dxf_file: str,
    frame_lineweight: int = 100,
    frame_color: int = _FRAME_COLOR,
    original_filename: Optional[str] = None,
) -> Dict:
    """通常モード（領域なし）用: 図面枠内ラベルから機器符号（候補）を抽出する。

    機器符号（候補）のうち確定パターンに一致するもの（`confirmed_labels`）は
    「未確定ラベル」UI でのレビューを経ずに自動採用される。それ以外
    （`review_labels`）はユーザーがチェックしたものだけが最終的な出力になる
    （既定モードでは初期状態は全て未選択）。

    戻り値 dict:
      filename: str
      confirmed_labels: [(ラベル, x, y), ...]（正規化済み、確定パターン一致＝自動採用）
      review_labels: [(ラベル, x, y), ...]（正規化済み、未確定ラベルUIでのレビュー対象）
      excluded_labels: [(ラベル, x, y), ...]（正規化済み、3パターンには一致したが
        除外パターンに該当し Reference Designator ではないと判断されたもの。
        「除外符号」シート用、2026-07-26 追加）
      total_in_frame: int
      warning: str | None （図面枠が見つからずフォールバックした場合等）
    """
    info = {
        'filename': original_filename or dxf_file,
        'confirmed_labels': [],
        'review_labels': [],
        'excluded_labels': [],
        'total_in_frame': 0,
        'warning': None,
    }

    collected = collect_in_frame_labels(dxf_file, frame_lineweight, frame_color)
    if collected['error']:
        # フォールバック自体は正常な抽出結果につながる（実際に正しく機器符号が
        # 抽出できるケースがほとんど）ため、「見つかりませんでした」という
        # 失敗を思わせる文言は避け、実施した内容を淡々と伝える言い回しにする
        # （ユーザー指摘、2026-07-14）。
        info['warning'] = (
            f'図面枠（太さ {frame_lineweight} の線で囲まれた枠）を検出できなかったため、'
            '図面枠内の制約なしに全ラベルを抽出しました。'
        )
        raw_labels = _collect_all_labels_fallback(dxf_file)
    else:
        raw_labels = collected['labels']

    normalized = normalize_labels(raw_labels)
    info['total_in_frame'] = len(normalized)
    candidates = classify_labels(normalized)
    info['confirmed_labels'], info['review_labels'] = split_confirmed(candidates)
    info['excluded_labels'] = classify_excluded_labels(normalized)
    return info


# ============================================================
# 7. 領域付きモード用: 領域名を付与した行データの構築
# ============================================================

def _aggregate_assigned_labels(assigned):
    """`assign_region_labels()` の出力 `[(text,x,y,names), ...]` から集計辞書を
    作る（`build_labeled_rows`〈named_regions 指定時〉と `build_region_output`
    で共用する自己完結的な集計ロジック）。

    戻り値: (cnt, region_of, in_region_count, label_count_per_region, region_label_counts)
      cnt: {ラベル: 出現数}
      region_of: {ラベル: {所属領域名, ...}}
      in_region_count: いずれかの領域に所属するラベルの出現件数
      label_count_per_region: {領域名: そのラベル出現数の合計}
      region_label_counts: {領域名: Counter({ラベル: 出現数})}
    """
    cnt = Counter()
    region_of = defaultdict(set)
    in_region_count = 0
    label_count_per_region = defaultdict(int)
    region_label_counts = defaultdict(Counter)
    for (text, _x, _y, names) in assigned:
        cnt[text] += 1
        if names:
            in_region_count += 1
        for n in names:
            region_of[text].add(n)
            label_count_per_region[n] += 1
            region_label_counts[n][text] += 1
    return cnt, region_of, in_region_count, label_count_per_region, region_label_counts


def build_labeled_rows(
    labels: List[Tuple[str, float, float]],
    named_regions: Optional[List[dict]] = None,
) -> List[dict]:
    """(text,x,y) リストから (ラベル,個数[,領域]) の行データを作る。

    named_regions が指定された場合は `assign_region_labels()` で領域名を
    割り当て、'領域' キー（カンマ区切り文字列）を各行に含める。
    """
    if named_regions is not None:
        assigned = assign_region_labels(labels, named_regions)
        cnt, region_of, _in_region_count, _label_count_per_region, _region_label_counts = \
            _aggregate_assigned_labels(assigned)
        return [
            {'ラベル': t, '個数': cnt[t], '領域': ', '.join(sorted(region_of[t]))}
            for t in sorted(cnt.keys())
        ]

    cnt = Counter(t for (t, _x, _y) in labels)
    return [{'ラベル': t, '個数': cnt[t]} for t in sorted(cnt.keys())]


def extract_ref_designator_rows(
    dxf_file: str,
    frame_lineweight: int = 100,
    frame_color: int = _FRAME_COLOR,
    original_filename: Optional[str] = None,
    named_regions: Optional[List[dict]] = None,
) -> Dict:
    """通常/領域付き両モード共用のトップレベル関数。

    図面枠内ラベルから機器符号（候補）の行データ（ラベル・個数[・領域]）を返す。
    named_regions を渡すと各行に '領域' キーが付与される（領域付きモード用）。

    戻り値 dict:
      filename, warning, total_in_frame: extract_ref_designator_data と同じ
      confirmed_rows: [{'ラベル':str,'個数':int[,'領域':str]}]（確定パターン一致＝自動採用）
      review_rows: [{'ラベル':str,'個数':int[,'領域':str]}]（未確定ラベルUIでのレビュー対象）
    """
    info = extract_ref_designator_data(dxf_file, frame_lineweight, frame_color, original_filename)
    return {
        'filename': info['filename'],
        'warning': info['warning'],
        'total_in_frame': info['total_in_frame'],
        'confirmed_rows': build_labeled_rows(info['confirmed_labels'], named_regions),
        'review_rows': build_labeled_rows(info['review_labels'], named_regions),
    }


def build_named_regions(
    analysis: dict,
    name_selections: dict,
    fname: str,
    start_no_name_idx: int = 0,
) -> Tuple[List[dict], int]:
    """領域付きモード用: 解析結果とユーザーが確定した名称選択から `assign_region_labels()`
    に渡す named リストを構築する（`region_detector.build_region_results()` の
    named 構築部分を機器符号（候補）パイプライン向けに複製したもの。
    region_detector.py は変更しないため、ここに独立して実装する）。

    戻り値: (named, next_no_name_idx)
    """
    named = []
    regions = analysis.get('regions', [])
    # 番号は複数の無名領域を区別するためだけのものなので、この図面に無名領域が
    # 1つしかない場合は付けない（ユーザー指摘: 番号があるとユーザーがその意味を
    # 理解できず不安に感じる。2026-07-23。region_detector.build_region_results()
    # と同じ方針）。
    total_no_name = sum(
        1 for reg in regions
        if not name_selections.get((fname, reg['id']), []) and not reg.get('name_candidates')
    )
    no_name_idx = start_no_name_idx
    for reg in regions:
        chosen_names = name_selections.get((fname, reg['id']), [])
        if not chosen_names and not reg.get('name_candidates'):
            no_name_idx += 1
            chosen_names = ["no name" if total_no_name == 1 else f"no name {no_name_idx}"]
        for nm in chosen_names:
            if not nm:
                continue
            named.append({
                'polygon': reg['polygon'], 'name': normalize_width(nm),
                'id': reg['id'], 'frame': reg['frame'], 'area_pct': reg['area_pct'],
            })
    return named, no_name_idx


def build_region_output(
    labels: List[Tuple[str, float, float]],
    named: List[dict],
    sort_value: str = 'asc',
) -> Dict:
    """(text,x,y) リストと named（`build_named_regions()` の出力）から、
    `create_region_excel_output()` に渡せる1ファイル分の集計結果を作る。

    `region_detector.build_region_results()` の集計部分（1ファイル分）を
    機器符号（候補）パイプライン向けに複製したもの（region_detector.py は
    変更しないため、ここに独立して実装する）。

    戻り値 dict: rows, named（label_count 付与済み）, in_region_count,
      region_label_counts
    """
    assigned = assign_region_labels(labels, named)
    cnt, region_of, in_region_count, label_count_per_region, region_label_counts = \
        _aggregate_assigned_labels(assigned)

    rows = [
        {'ラベル': t, '個数': cnt[t], '領域': ', '.join(sorted(region_of[t]))}
        for t in cnt
    ]
    if sort_value == 'asc':
        rows.sort(key=lambda r: r['ラベル'])
    elif sort_value == 'desc':
        rows.sort(key=lambda r: r['ラベル'], reverse=True)

    for r in named:
        r['label_count'] = label_count_per_region.get(r['name'], 0)

    return {
        'rows': rows,
        'named': named,
        'in_region_count': in_region_count,
        'region_label_counts': {n: dict(c) for n, c in region_label_counts.items()},
    }


def _approved_final_labels(data: dict, approved: set) -> list:
    """`ref_pending[fname]`（確定分＋未確定ラベルUIレビュー対象）と、ユーザーが
    採用したラベル文字列の集合から、最終的に出力する (text,x,y) リストを返す
    （`build_region_results_from_pending`/`build_ref_final_from_pending` 共用）。"""
    return list(data['confirmed_labels']) + [
        item for item in data['review_labels'] if item[0] in approved
    ]


def build_region_results_from_pending(
    ref_pending: dict,
    region_analyses: dict,
    name_selections_by_file: dict,
    approved_by_file: dict,
    sort_value: str = 'asc',
) -> dict:
    """「選択完了」時（領域付きモード）の集計。`ref_pending`（未確定ラベルUI
    表示用データ）と `region_analyses`（領域検出結果）から、
    `create_region_excel_output()` に渡せる region_results を構築する。

    `name_selections_by_file`: `{fname: {(fname, reg_id): [name, ...]}}`
      （`app.py` の `_gather_name_selections(fname, analysis)` の結果を
      ファイルごとに集めたもの。チェックボックスの選択状態＝UI依存のため、
      この関数自身は計算せず呼び出し側から受け取るだけ）。
    `approved_by_file`: `{fname: {採用されたラベル文字列, ...}}`
    """
    region_results = {}
    for fname, data in ref_pending.items():
        approved = approved_by_file[fname]
        final_labels = _approved_final_labels(data, approved)
        analysis = region_analyses[fname]
        named, _ = build_named_regions(analysis, name_selections_by_file[fname], fname)
        out = build_region_output(final_labels, named, sort_value)
        review_texts = {t for t, _x, _y in data['review_labels']}
        region_results[fname] = {
            'rows': out['rows'],
            'named': out['named'],
            'frames': len(analysis.get('frames', [])),
            'regions_detected': len(analysis.get('regions', [])),
            'regions_named': len({r['id'] for r in named}),
            'total_in_frame': data['total_in_frame'],
            'filtered_count': len(review_texts - approved),
            'final_count': len(final_labels),
            'in_region_count': out['in_region_count'],
            'drawing_number': analysis.get('main_drawing_number') or '',
            'title': analysis.get('title'),
            'region_label_counts': out['region_label_counts'],
            'excluded_rows': build_labeled_rows(data.get('excluded_labels', [])),
        }
    return region_results


def build_ref_final_from_pending(
    ref_pending: dict,
    approved_by_file: dict,
    sort_value: str = 'asc',
) -> dict:
    """「選択完了」時（通常モード）の集計。`ref_pending` から
    `create_ref_designator_excel_output()` に渡せる ref_final を構築する。"""
    ref_final = {}
    for fname, data in ref_pending.items():
        approved = approved_by_file[fname]
        final_labels = _approved_final_labels(data, approved)
        rows = build_labeled_rows(final_labels)
        if sort_value == 'desc':
            rows.sort(key=lambda r: r['ラベル'], reverse=True)
        review_texts = {t for t, _x, _y in data['review_labels']}
        ref_final[fname] = {
            'rows': rows,
            'total_in_frame': data['total_in_frame'],
            'unclassified_count': len(review_texts - approved),
            'warning': data.get('warning'),
            'main_drawing_number': data.get('main_drawing_number'),
            'source_drawing_number': data.get('source_drawing_number'),
            'title': data.get('title'),
            'subtitle': data.get('subtitle'),
            'excluded_rows': build_labeled_rows(data.get('excluded_labels', [])),
        }
    return ref_final
