"""機器符号（候補）抽出モジュール（v1.6.0）

reference_designator_candidates.xlsx（`Patterns` / `ExclusionPatterns` シート）を
正としてパターン・除外リストを実装する。DXF-extract-labels 固有の新機能であり、
他プロジェクトとの共有コピーは無い（`utils/extract_labels.py` とは独立に保つ）。

処理の流れ:
  1. 図面枠（lineweight=frame_lineweight かつ color=7 の LINE 4本で構成）を検出
  2. 図面枠を構成するブロック（フォーマットブロック。図面情報欄・枠外位置記号
     A-F/1-8 を含む）由来の TEXT/MTEXT を丸ごと除外
  3. 図面枠内に残ったラベルを NFKC 正規化し、括弧より前の部分でパターン判定
  4. 3パターンのいずれかに一致し、かつ除外パターンに該当しないものを
     「機器符号（候補）」、それ以外を「未確定ラベル」とする
"""
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import ezdxf

from .common_utils import normalize_width
from .extract_labels import extract_text_from_entity, _block_has_text_content
from .region_detector import detect_drawing_frames, assign_region_labels


# ============================================================
# 1. Reference Designator パターン（Patterns シートが正）
# ============================================================

_PATTERN_CORE = (
    r'[A-Z]+-[A-Z]+[0-9]+[A-Z0-9-]*'   # 英字繰返し-英字繰返し+数字繰返し+英数字/ハイフン任意
    r'|[A-Z]+[0-9]+[A-Z0-9-]*'          # 英字繰返し+数字繰返し+英数字/ハイフン任意
    r'|[A-Z]+'                          # 英字繰返しのみ
)
CANDIDATE_PATTERN = re.compile(r'^(?:%s)$' % _PATTERN_CORE)


# ============================================================
# 2. 除外パターン（ExclusionPatterns シートが正、2026-07-10 確定）
# ============================================================

EXCLUSION_REGEXES = [
    re.compile(r'^[A-Z]$'),                                          # single_letter_position
    re.compile(r'.*[+-]$'),                                          # trailing_sign
    re.compile(r'^AWG[0-9]*$'),                                      # wire_gauge
    re.compile(r'^RACK[0-9]*(-[0-9]+)?$'),                           # rack_prefix
    re.compile(r'^[A-Z]{2}[0-9]{4}-[0-9]{3}(-[0-9]{2})?[A-Z]?$'),    # drawing_number
    re.compile(r'^(JIS|DWG)[A-Z0-9]*$'),                             # jis_dwg_prefix
    re.compile(r'^[AB][0-9]+$'),                                     # terminal_row_letter_digit
    re.compile(r'^PE[0-9]+$'),                                       # earth_terminal_digit
    re.compile(r'^[LNP][0-9]+[A-Z]?$'),                              # phase_rail_letter_digit
    re.compile(r'^X[A-Z]+$'),                                        # io_signal_x_prefix
]

_COMMON_NOUNS = {
    'ABORT', 'ACCESSORY', 'ALARM', 'ANNEAL', 'ANODE', 'AUTO', 'AUTOSTART',
    'BRAKE', 'BUSY', 'BUZZER', 'BYPASS', 'CATHODE', 'CHAMBER', 'CHANGE',
    'CHILLER', 'CIRCUIT', 'CLOSE', 'COLD', 'CONTACT', 'CONTROL',
    'CONTROLLER', 'COVER', 'CPU', 'DATA', 'DETECT', 'DEVICENET', 'DRAIN',
    'ENABLE', 'ENCODER', 'ETHERCAT', 'ETHERNET', 'EXHAUST', 'EXTEND',
    'FAIL', 'FLOW', 'FREE', 'FUNCTION', 'HDMI', 'HOST', 'HOT', 'INPUT',
    'INTELOCK', 'INTERFACE', 'INTERLOCK', 'KEYBOARD', 'KEYBORD', 'LABEL',
    'LINE', 'LOCK', 'MASTER', 'MODE', 'MODULE', 'MONITOR', 'MOTOR',
    'MOUSE', 'MOVE', 'NC', 'NEG', 'NETWORK', 'NO', 'NOTE', 'NPN', 'OPEN',
    'OUTPUT', 'PANEL', 'PARAMETER', 'PLC', 'PNP', 'POS', 'POSITION',
    'PRESET', 'PRESSURE', 'PULS', 'RDY', 'RECALL', 'RECEPTACLE', 'RELAY',
    'RELEASE', 'REMOTE', 'RESET', 'RETRACT', 'RUN', 'SELECT', 'SENSOR',
    'SERIAL', 'SERVICE', 'SET', 'SETTING', 'SHUTTER', 'SIGN', 'SLAVE',
    'SLOT', 'SPARE', 'START', 'STATAUS', 'STATUS', 'STO', 'STOP',
    'SWITCH', 'SYSTEM', 'TERMINAL', 'THERMOCOUPLE', 'TIME', 'TRIGGER',
    'USB', 'VGA', 'VIDEO', 'WATCHDOG', 'WATER', 'WIRING', 'ZERO',
}

_CIRCUIT_DESCRIPTION = {
    'AC', 'ACIN', 'AGND', 'COM', 'DC', 'DCIN', 'FG', 'GND', 'IN', 'LOAD',
    'OFF', 'ON', 'OUT', 'PE', 'PGND', 'POW', 'POWER', 'POWIN', 'PWR',
    'SG', 'VAC', 'VCC', 'VDC',
}

_UNIT_NAMES = {
    'CASE', 'CTC', 'EFEM', 'FOUP', 'LA', 'LB', 'LINEA', 'LINEB', 'LL',
    'SH', 'SHIELD', 'TM',
}

_CABLE_COLORS = {
    'BK', 'BL', 'BLACK', 'BLK', 'BLU', 'BLUE', 'BR', 'BRN', 'BROWN', 'GN',
    'GNYE', 'GRAY', 'GREEN', 'GREY', 'GRN', 'GY', 'OR', 'ORANGE', 'PINK',
    'PK', 'PU', 'PURPLE', 'RD', 'RED', 'SB', 'VIOLET', 'VT', 'WH',
    'WHITE', 'YE', 'YELLOW',
}

_TITLEBLOCK_TERMS = {
    'ANGLE', 'APPROVED', 'APPRV', 'CHECK', 'CHECKED', 'DATE', 'DESIG',
    'DESIGNED', 'DRAW', 'DRAWN', 'FINISH', 'ISSUED', 'MARK', 'MATERIAL',
    'NAME', 'REMARKS', 'REV', 'REVISION', 'SCALE', 'SHEET', 'SIZE',
    'TITLE', 'TOLERANCES', 'UNIT', 'WEIGHT',
}
# スペース/ピリオドを含む語句（UNLESS NOTED, MFG No. 等）は CANDIDATE_PATTERN
# （英大文字・数字・ハイフンのみ）に元々一致しないため EXCLUSION_EXACT に含める
# 必要はない（候補にすらならない）。図面情報枠の構造的除外（フォーマット
# ブロック丸ごと除外）が第一防衛線であり、本リストは第二防衛線。

EXCLUSION_EXACT = (
    _COMMON_NOUNS | _CIRCUIT_DESCRIPTION | _UNIT_NAMES | _CABLE_COLORS
    | _TITLEBLOCK_TERMS
)


def normalize_label(label: str) -> str:
    """NFKC正規化+前後空白除去した表示用ラベルを返す（括弧は保持）。"""
    if not label:
        return ''
    return unicodedata.normalize('NFKC', label).strip()


def _judgment_text(normalized_label: str) -> str:
    """判定用文字列を返す（括弧以降を除く）。例: 'R10(2.2K)' -> 'R10'。"""
    idx = normalized_label.find('(')
    return normalized_label[:idx] if idx >= 0 else normalized_label


def _classify_judgment(judgment: str) -> str:
    """判定用文字列（括弧より前）を 'candidate' / 'excluded' / 'no_match' に分類する。

    'no_match' は3パターン（Patterns シート）のいずれにも一致しない文字列
    （説明文・記号・注記等）。reference_designator_candidates.xlsx の
    ReferenceDesignators シート（Patterns 一致のみを集めたもの）に元々含まれない
    ため、機器符号（候補）にも未確定ラベルにも分類しない（2026-07-10 ユーザー指摘。
    未確定ラベルは「Patterns には一致したが除外パターンに該当したもの」＝
    RemainingUnclassified シートと同じ母集団に限る）。
    """
    if not judgment or not CANDIDATE_PATTERN.match(judgment):
        return 'no_match'
    if judgment in EXCLUSION_EXACT:
        return 'excluded'
    if any(rx.match(judgment) for rx in EXCLUSION_REGEXES):
        return 'excluded'
    return 'candidate'


def is_ref_designator_candidate(label: str) -> bool:
    """正規化済みラベル（表示用、括弧を含みうる）が機器符号（候補）かどうかを返す。

    判定は括弧より前の部分に対して行う。呼び出し側は `normalize_label()` で
    正規化した文字列を渡すこと（内部では再正規化しない）。
    """
    return _classify_judgment(_judgment_text(label)) == 'candidate'


def split_candidates(
    labels: List[str],
) -> Tuple[List[str], List[str]]:
    """正規化済みラベルのリストを (機器符号候補, 未確定ラベル) に分ける。

    未確定ラベルは Patterns（3パターン）に一致し除外パターンに該当したものだけ。
    3パターンいずれにも一致しない文字列はどちらにも含めず捨てる
    （`_classify_judgment` 参照）。
    """
    candidates = []
    unclassified = []
    for label in labels:
        status = _classify_judgment(_judgment_text(label))
        if status == 'candidate':
            candidates.append(label)
        elif status == 'excluded':
            unclassified.append(label)
    return candidates, unclassified


def summarize_labels(labels: List[str]) -> List[Tuple[str, int]]:
    """ラベルリストを (ラベル, 個数) にカウントし、ラベル昇順で返す。"""
    counter = Counter(labels)
    return [(lbl, counter[lbl]) for lbl in sorted(counter.keys())]


# ============================================================
# 3. 図面枠検出・フォーマットブロック（図面情報欄）の構造的除外
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

    戻り値: (frame_lines, label_entities)
      frame_lines: [(start, end), ...]（Vec3のまま。detect_drawing_frames に渡す）
      label_entities: TEXT/MTEXT エンティティのリスト（フォーマットブロック由来を除く）
    """
    msp = doc.modelspace()
    fmt_blocks = _format_block_names(doc, frame_lineweight, frame_color)

    frame_lines = []
    label_entities = []
    text_block_cache = {}

    def is_frame_line(e):
        return (getattr(e.dxf, 'lineweight', None) == frame_lineweight
                and getattr(e.dxf, 'color', None) == frame_color)

    for e in msp:
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
    ファイル全体（modelspace）のラベルを収集する。"""
    try:
        doc = ezdxf.readfile(dxf_file)
    except Exception:
        return []
    msp = doc.modelspace()
    text_block_cache = {}
    out = []
    for e in msp:
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


# ============================================================
# 4. ラベル正規化・機器符号候補/未確定ラベルへの分類（座標付き）
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
) -> Tuple[List[Tuple[str, float, float]], List[Tuple[str, float, float]]]:
    """正規化済み (text,x,y) リストを (機器符号候補, 未確定ラベル) に分ける。

    未確定ラベルは Patterns（3パターン）に一致し除外パターンに該当したものだけ
    （reference_designator_candidates.xlsx の RemainingUnclassified シートと同じ
    母集団＝ ReferenceDesignators〔Patterns一致〕から除外分類されたもの、の
    "裏側"＝除外された側）。3パターンいずれにも一致しない文字列（説明文・記号等、
    例 `(2/5)`）はどちらにも分類せず捨てる（2026-07-10 ユーザー指摘）。
    """
    candidates = []
    unclassified = []
    for item in labels:
        status = _classify_judgment(_judgment_text(item[0]))
        if status == 'candidate':
            candidates.append(item)
        elif status == 'excluded':
            unclassified.append(item)
    return candidates, unclassified


# ============================================================
# 5. 通常モード（領域なし）用トップレベル関数
# ============================================================

def extract_ref_designator_data(
    dxf_file: str,
    frame_lineweight: int = 100,
    frame_color: int = _FRAME_COLOR,
    original_filename: Optional[str] = None,
) -> Dict:
    """通常モード（領域なし）用: 図面枠内ラベルを機器符号候補/未確定に分ける。

    戻り値 dict:
      filename: str
      candidate_labels: [(ラベル, x, y), ...]（正規化済み、機器符号候補）
      unclassified_labels: [(ラベル, x, y), ...]（正規化済み、未確定）
      total_in_frame: int
      warning: str | None （図面枠が見つからずフォールバックした場合等）
    """
    info = {
        'filename': original_filename or dxf_file,
        'candidate_labels': [],
        'unclassified_labels': [],
        'total_in_frame': 0,
        'warning': None,
    }

    collected = collect_in_frame_labels(dxf_file, frame_lineweight, frame_color)
    if collected['error']:
        info['warning'] = collected['error'] + '（図面枠内フィルタなしで全ラベルを対象にします）'
        raw_labels = _collect_all_labels_fallback(dxf_file)
    else:
        raw_labels = collected['labels']

    normalized = normalize_labels(raw_labels)
    info['total_in_frame'] = len(normalized)

    candidates, unclassified = classify_labels(normalized)
    info['candidate_labels'] = candidates
    info['unclassified_labels'] = unclassified
    return info


# ============================================================
# 6. 領域付きモード用: 領域名を付与した行データの構築
# ============================================================

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
        cnt = Counter()
        region_of = defaultdict(set)
        for (text, _x, _y, names) in assigned:
            cnt[text] += 1
            for n in names:
                region_of[text].add(n)
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

    図面枠内ラベルを機器符号候補/未確定の行データ（ラベル・個数[・領域]）に分けて返す。
    named_regions を渡すと各行に '領域' キーが付与される（領域付きモード用）。

    戻り値 dict:
      filename, warning, total_in_frame: extract_ref_designator_data と同じ
      candidate_rows / unclassified_rows: [{'ラベル':str,'個数':int[,'領域':str]}]
    """
    info = extract_ref_designator_data(dxf_file, frame_lineweight, frame_color, original_filename)
    return {
        'filename': info['filename'],
        'warning': info['warning'],
        'total_in_frame': info['total_in_frame'],
        'candidate_rows': build_labeled_rows(info['candidate_labels'], named_regions),
        'unclassified_rows': build_labeled_rows(info['unclassified_labels'], named_regions),
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
    no_name_idx = start_no_name_idx
    for reg in analysis.get('regions', []):
        chosen_names = name_selections.get((fname, reg['id']), [])
        if not chosen_names and not reg.get('name_candidates'):
            no_name_idx += 1
            chosen_names = [f"no name {no_name_idx}"]
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
