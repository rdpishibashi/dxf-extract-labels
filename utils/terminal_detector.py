"""端子台(TB)ラベル・矩形検出モジュール（DXF-extract-labels 専用）

「UNIT内結線図」というタイトルの図面から、"TB" で始まり英大文字・数字が
1文字以上続くラベル（端子台名候補、例: `TB001`・`TBN241`）に対応する端子台
矩形を特定し、矩形内に記載された端子番号（数字のみのラベル）を抽出する。

識別キー:
  - 矩形の辺   : lineweight=25 かつ color=2(ACI黄) の LINE
                （矩形領域抽出＝`region_detector.py` の領域境界線と同じ識別キー）
  - 端子(ギャップ補完): lineweight=50 かつ color=4(ACI水色) の CIRCLE。
    部品（端子記号）が辺の途中に配置されて LINE を途切れさせている箇所を、
    この CIRCLE が橋渡しする。**CIRCLE による橋渡しが1箇所以上ある矩形のみを
    端子台矩形として扱う**（LINEのみで閉じた矩形は、矩形領域抽出の領域境界線
    そのものである可能性があり判別できないため、対象外とする。ユーザー確認済み）。

対象ファイル判定:
  タイトル（NFKC正規化後）に「UNIT内結線図」を含み、かつサブタイトルが
  「TB COMPONENT」でないこと。「TB COMPONENT」は端子台の全端子番号一覧ページ
  であり、そこに現れる数字は「使用中の結線」ではなく「端子台の全端子番号」の
  ため意味が異なる（ユーザー確認済み、除外対象）。

ラベル-矩形の対応判定:
  マージン（距離のしきい値）をオプションとして公開せず、常に矩形を構造的に
  探索して対応付ける（ユーザー指示）。ラベルは矩形の四辺のうちいずれかの
  辺の近傍にあり、実データでは「矩形の直上」（最頻出）と「矩形の直下」の
  両方が観測されている。同一の物理位置に複数の矩形が隣接して存在する場合
  （例: 1つのラベルが上の矩形の下辺にも下の矩形の上辺にも近い）に誤対応
  しないよう、「ラベルが矩形の上（primary）」を「ラベルが矩形の下
  （secondary）」より常に優先し、全ラベル・全矩形の候補ペアを
  (優先度, 距離) 昇順でグリーディに1:1マッチングする
  （`EE6492-039-38A.dxf` の TBP044/TBN241 隣接矩形の衝突で確認・解消済み）。

  図面全体が90°回転している場合は「上/下」が「右端/左端」に入れ替わる
  （`region_detector._rotated_edge_roles` と同じ規約: 回転角+90°多数派なら
  primary=右端、-90°多数派なら primary=左端）。**実データに90°回転した
  UNIT内結線図のサンプルが無いため、この分岐は合成DXFの単体テストのみで
  検証している。実サンプルが見つかった場合は方向の妥当性を再確認すること。**

候補ラベルの判定:
  「TB」で始まり、その直後に英大文字・数字が1文字以上続くもの（正規表現
  `^TB[A-Z0-9]+`、NFKC正規化後の文字列に対して先頭一致で判定。後続に続く
  文字列〔スペース・カッコ・電流値等〕は問わない）。この条件により、
  `TB取付板`（TBの直後が漢字）や `TB COMPONENT`（TBの直後がスペース）を
  構造的に候補から除外する（ユーザー確認済み）。

TB List シートの集計（`build_terminal_rows`）:
  「端子台」（ラベルの正規化済みテキスト）でユニークをとり、複数ファイルに
  またがって同じ端子台名が登場する場合は、端子番号・図番を統合して1行に
  まとめる。「端子No.」は同じ番号が複数回登場する場合 `N(件数)` の形式で
  表示し、そうでなければ素の数値を表示する（例: `1, 2, 3, 7(2)`）。図番は
  複数ある場合カンマ区切りでABC順に列挙する。行自体も「端子台」のABC順。

  候補パターンには一致したが対応する矩形が見つからなかったラベル
  （`unmatched_labels`）は、末尾に1行の空行を挟んだうえで「端子台」列に
  `端子検出不可`、「端子No.」列にそのラベルテキスト、「図番」列にその
  ファイルの図番を記載する（ユーザー指定の仕様）。
"""
import os
import re
from collections import Counter, defaultdict

import ezdxf

from .extract_labels import extract_text_from_entity, extract_labels
from .common_utils import normalize_width
from .region_detector import _label_rotation_angle


TB_LINE_WEIGHT = 25
TB_LINE_COLOR = 2
TB_CIRCLE_WEIGHT = 50
TB_CIRCLE_COLOR = 4

_ENDPOINT_TOL = 0.15        # 同一点とみなす座標許容誤差
_BRIDGE_AXIS_TOL = 0.5      # 円が橋渡しする軸(x or y)の一致許容誤差
_BRIDGE_SEARCH_RANGE = 15.0  # 円から橋渡し相手のLINE端点を探す最大距離
_MATCH_MAX_DIST = 10.0      # ラベル-矩形辺間のマッチとして許容する最大距離
_MATCH_XY_TOL = 1.0         # ラベルが矩形の幅(高さ)範囲内とみなす許容誤差
_MIN_RECT_SIZE = 1.0        # 矩形として扱う最小の幅・高さ
_ROTATION_THRESHOLD = 0.5   # 図面回転判定の閾値（ラベルの過半数が90°回転で「回転」）

# 候補ラベル: "TB" + 英大文字/数字の繰り返し（1文字以上）。後続の文字列は問わない。
_TB_LABEL_PATTERN = re.compile(r'^TB[A-Z0-9]+')


class _UnionFind:
    """端子台矩形の辺(LINE)端点を連結成分にまとめるための Union-Find。"""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _collect_geometry(doc):
    """Model Space を基本に、INSERT展開を含めてラベル(テキスト+回転角)・
    矩形辺LINE・橋渡しCIRCLEを収集する。Model Space に何の内容も無い場合に
    限り、Model 以外のレイアウト（Paper Space等）を順に試す。**同一レイアウト
    内の texts/lines/circles のみを組み合わせて返す**（レイアウトをまたいで
    混在させない）。

    Model Space と Paper Space は完全に独立した座標系であり、一方のレイアウト
    のラベルと別レイアウトの矩形辺LINEを組み合わせると、TBラベルと矩形の
    距離ベースの対応判定が座標的に成立しなくなる（`region_detector.py`・
    `ref_designator.py` で確認・修正済みの不具合と同根。2026-07-14）。
    """
    def collect_from_layout(layout):
        texts = []      # (text, x, y, rotation_deg)
        lines = []      # (x1, y1, x2, y2)
        circles = []    # (cx, cy, r)

        def handle_entity(e):
            t = e.dxftype()
            if t in ('TEXT', 'MTEXT'):
                _raw, cleaned, (x, y) = extract_text_from_entity(e)
                if cleaned:
                    texts.append((cleaned, x, y, _label_rotation_angle(e)))
            elif t == 'LINE':
                if (getattr(e.dxf, 'lineweight', None) == TB_LINE_WEIGHT
                        and getattr(e.dxf, 'color', None) == TB_LINE_COLOR):
                    s, en = e.dxf.start, e.dxf.end
                    if (s[0], s[1]) != (en[0], en[1]):
                        lines.append((s[0], s[1], en[0], en[1]))
            elif t == 'CIRCLE':
                if (getattr(e.dxf, 'lineweight', None) == TB_CIRCLE_WEIGHT
                        and getattr(e.dxf, 'color', None) == TB_CIRCLE_COLOR):
                    c = e.dxf.center
                    circles.append((c[0], c[1], e.dxf.radius))

        for e in layout:
            if e.dxftype() == 'INSERT':
                try:
                    for v in e.virtual_entities():
                        handle_entity(v)
                except Exception:
                    pass
            else:
                handle_entity(e)

        return texts, lines, circles

    result = collect_from_layout(doc.modelspace())
    if any(result):
        return result

    try:
        for layout in doc.layouts:
            if layout.name != 'Model':
                alt_result = collect_from_layout(layout)
                if any(alt_result):
                    return alt_result
    except Exception:
        pass

    return result


def _detect_rotation_mode(texts, threshold=_ROTATION_THRESHOLD):
    """ラベル回転角の多数派から 'right'/'left'/None を判定する
    （`region_detector._rotated_edge_roles` と同じ規約・実装方針）。"""
    total = 0
    near_plus90 = 0
    near_minus90 = 0
    for _t, _x, _y, ang in texts:
        total += 1
        norm = ((ang + 180.0) % 360.0) - 180.0
        if 80.0 <= norm <= 100.0:
            near_plus90 += 1
        elif -100.0 <= norm <= -80.0:
            near_minus90 += 1
    if total == 0:
        return None
    if (near_plus90 / total) >= threshold:
        return 'right'
    if (near_minus90 / total) >= threshold:
        return 'left'
    return None


def _build_rect_candidates(lines, circles, endpoint_tol=_ENDPOINT_TOL,
                            bridge_axis_tol=_BRIDGE_AXIS_TOL,
                            bridge_search_range=_BRIDGE_SEARCH_RANGE,
                            min_size=_MIN_RECT_SIZE):
    """LINE(辺)とCIRCLE(橋渡し)から、端子台矩形候補のリストを構築する。

    各候補は {'xl','xr','y0','y1'} の bbox。CIRCLE橋渡しが1箇所も無い連結成分、
    および4辺すべてが揃っていない連結成分は候補から除外する。
    """
    endpoints = []  # 全LINE端点 [(x, y), ...]（偶数idx=始点、奇数idx=終点）
    for (x1, y1, x2, y2) in lines:
        endpoints.append((x1, y1))
        endpoints.append((x2, y2))

    def endpoint_key(idx):
        line_idx, which = divmod(idx, 2)
        return ('L', line_idx, which)

    uf = _UnionFind()

    # 同一LINEの2端点を結合
    for i in range(len(lines)):
        uf.union(endpoint_key(2 * i), endpoint_key(2 * i + 1))

    # 近傍端点同士をグリッドバケットで結合（O(n)相当）
    cell = max(endpoint_tol * 2, 0.5)
    grid = defaultdict(list)

    def cell_key(pt):
        return (round(pt[0] / cell), round(pt[1] / cell))

    for idx, pt in enumerate(endpoints):
        grid[cell_key(pt)].append(idx)

    for idx, pt in enumerate(endpoints):
        cx, cy = cell_key(pt)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other_idx in grid.get((cx + dx, cy + dy), ()):
                    if other_idx <= idx:
                        continue
                    other = endpoints[other_idx]
                    if (abs(other[0] - pt[0]) <= endpoint_tol
                            and abs(other[1] - pt[1]) <= endpoint_tol):
                        uf.union(endpoint_key(idx), endpoint_key(other_idx))

    # CIRCLE橋渡し: 円中心と同一x(縦)/同一y(横)の端点のうち、円を挟んで
    # 両側にある最近傍の対を結合する。
    bridged_keys = set()
    for (cx, cy, _r) in circles:
        vert_above, vert_below = None, None
        horiz_left, horiz_right = None, None
        for idx, (px, py) in enumerate(endpoints):
            if abs(px - cx) <= bridge_axis_tol and abs(py - cy) <= bridge_search_range:
                if py > cy and (vert_above is None or py < endpoints[vert_above][1]):
                    vert_above = idx
                elif py < cy and (vert_below is None or py > endpoints[vert_below][1]):
                    vert_below = idx
            if abs(py - cy) <= bridge_axis_tol and abs(px - cx) <= bridge_search_range:
                if px < cx and (horiz_left is None or px > endpoints[horiz_left][0]):
                    horiz_left = idx
                elif px > cx and (horiz_right is None or px < endpoints[horiz_right][0]):
                    horiz_right = idx

        if vert_above is not None and vert_below is not None:
            a_key, b_key = endpoint_key(vert_above), endpoint_key(vert_below)
            uf.union(a_key, b_key)
            bridged_keys.add(a_key)
            bridged_keys.add(b_key)
        if horiz_left is not None and horiz_right is not None:
            l_key, r_key = endpoint_key(horiz_left), endpoint_key(horiz_right)
            uf.union(l_key, r_key)
            bridged_keys.add(l_key)
            bridged_keys.add(r_key)

    # 連結成分ごとに集計
    components = defaultdict(list)
    for idx in range(len(endpoints)):
        components[uf.find(endpoint_key(idx))].append(idx)

    tol_side = max(endpoint_tol, 0.5)
    rects = []
    for _root, idxs in components.items():
        has_bridge = any(endpoint_key(i) in bridged_keys for i in idxs)
        if not has_bridge:
            continue

        pts = [endpoints[i] for i in idxs]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        xl, xr, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        if xr - xl < min_size or y1 - y0 < min_size:
            continue

        member_line_idxs = {i // 2 for i in idxs}

        def has_side(pred, _lines_idx=member_line_idxs):
            return any(pred(lines[li]) for li in _lines_idx)

        left_ok = has_side(lambda l: abs(l[0] - xl) <= tol_side and abs(l[2] - xl) <= tol_side
                            and abs(l[1] - l[3]) > tol_side)
        right_ok = has_side(lambda l: abs(l[0] - xr) <= tol_side and abs(l[2] - xr) <= tol_side
                             and abs(l[1] - l[3]) > tol_side)
        bottom_ok = has_side(lambda l: abs(l[1] - y0) <= tol_side and abs(l[3] - y0) <= tol_side
                              and abs(l[0] - l[2]) > tol_side)
        top_ok = has_side(lambda l: abs(l[1] - y1) <= tol_side and abs(l[3] - y1) <= tol_side
                           and abs(l[0] - l[2]) > tol_side)

        if left_ok and right_ok and bottom_ok and top_ok:
            rects.append({'xl': xl, 'xr': xr, 'y0': y0, 'y1': y1})

    return rects


def _label_rect_distances(rotated_mode, lx, ly, xl, xr, y0, y1, xy_tol):
    """ラベル(lx,ly)と矩形の (primary_dist, secondary_dist) を返す。
    どちらも矩形の外側（primary/secondary いずれかが必ず非負）を想定し、
    直交軸方向が矩形の範囲内(±xy_tol)に無ければ (None, None) を返す。"""
    if rotated_mode == 'right':
        if not (y0 - xy_tol <= ly <= y1 + xy_tol):
            return None, None
        return lx - xr, xl - lx
    if rotated_mode == 'left':
        if not (y0 - xy_tol <= ly <= y1 + xy_tol):
            return None, None
        return xl - lx, lx - xr
    if not (xl - xy_tol <= lx <= xr + xy_tol):
        return None, None
    return ly - y1, y0 - ly


def _match_labels_to_rects(tb_labels, rects, rotated_mode,
                            xy_tol=_MATCH_XY_TOL, max_dist=_MATCH_MAX_DIST):
    """TBラベルと矩形候補を1:1でグリーディにマッチングする。

    「ラベルが矩形の上(primary)」を「ラベルが矩形の下(secondary)」より
    常に優先し、(優先度, 距離) 昇順の候補リストから貪欲に割り当てる
    （隣接する2矩形が1ラベルに対し両方距離的に近い場合の衝突を、
    大域的な優先度で解消する。`EE6492-039-38A.dxf` で確認済み）。

    戻り値: (matches: {label_idx: rect_idx}, unmatched_label_idxs: [int])
    """
    candidates = []
    for li, (_t, lx, ly, _rot) in enumerate(tb_labels):
        for ri, r in enumerate(rects):
            primary, secondary = _label_rect_distances(
                rotated_mode, lx, ly, r['xl'], r['xr'], r['y0'], r['y1'], xy_tol)
            if primary is None:
                continue
            if 0 <= primary <= max_dist:
                candidates.append((0, primary, li, ri))
            elif 0 <= secondary <= max_dist:
                candidates.append((1, secondary, li, ri))

    candidates.sort(key=lambda c: (c[0], c[1]))

    matches = {}
    used_rects = set()
    for _prio, _dist, li, ri in candidates:
        if li in matches or ri in used_rects:
            continue
        matches[li] = ri
        used_rects.add(ri)

    unmatched = [i for i in range(len(tb_labels)) if i not in matches]
    return matches, unmatched


def _collect_digits(texts, rect, pad=0.5):
    """矩形内(パディング込み)にある数字のみのラベルを整数リストとして返す。"""
    xl, xr = rect['xl'] - pad, rect['xr'] + pad
    y0, y1 = rect['y0'] - pad, rect['y1'] + pad
    nums = []
    for (t, x, y, _rot) in texts:
        if not (xl <= x <= xr and y0 <= y <= y1):
            continue
        norm = normalize_width(t).strip()
        if norm.isdigit():
            nums.append(int(norm))
    return nums


def analyze_dxf_terminals(dxf_path, original_filename=None):
    """DXFファイルから端子台情報を抽出する。

    Returns:
        dict:
            'is_target': 対象ファイル（タイトルが「UNIT内結線図」を含み、
                サブタイトルが「TB COMPONENT」でない）かどうか
            'title' / 'subtitle': 抽出結果（対象外でも可能な範囲で埋める）
            'drawing_number': 図番（未抽出時はファイル名〔拡張子なし〕）
            'entries': [{'label': str, 'numbers': [int, ...]}, ...]
                （ファイル内で同名ラベルの端子番号を集約・昇順ソート済み）
            'unmatched_labels': 対応する矩形が見つからなかったTBラベル
                （正規化・重複除去・ソート済み）
            'error': 例外発生時のメッセージ（正常時は None）
    """
    original_filename = original_filename or os.path.basename(dxf_path)
    result = {
        'is_target': False,
        'title': None,
        'subtitle': None,
        'drawing_number': None,
        'entries': [],
        'unmatched_labels': [],
        'error': None,
    }

    try:
        _labels, info = extract_labels(
            dxf_path,
            extract_title_option=True,
            extract_drawing_numbers_option=True,
            original_filename=original_filename,
        )
    except Exception as e:
        result['error'] = str(e)
        return result

    title = info.get('title')
    subtitle = info.get('subtitle')
    result['title'] = title
    result['subtitle'] = subtitle
    result['drawing_number'] = (
        info.get('main_drawing_number') or os.path.splitext(original_filename)[0]
    )

    title_norm = normalize_width(title or '')
    if 'UNIT内結線図' not in title_norm:
        return result

    subtitle_norm = normalize_width(subtitle or '').strip()
    if subtitle_norm == 'TB COMPONENT':
        return result

    result['is_target'] = True

    try:
        doc = ezdxf.readfile(dxf_path)
    except Exception as e:
        result['error'] = str(e)
        return result

    texts, lines, circles = _collect_geometry(doc)
    tb_labels = [
        (t, x, y, rot) for (t, x, y, rot) in texts
        if _TB_LABEL_PATTERN.match(normalize_width(t))
    ]
    rects = _build_rect_candidates(lines, circles)
    rotated_mode = _detect_rotation_mode(texts)

    matches, unmatched_idx = _match_labels_to_rects(tb_labels, rects, rotated_mode)

    # 重複（同一ラベルが複数矩形にまたがり同じ番号を含む等）を保持したまま集める。
    # 出現回数を反映した表示（'N(件数)'）は build_terminal_rows() 側で行う。
    grouped = defaultdict(list)
    for label_idx, rect_idx in matches.items():
        label_text = normalize_width(tb_labels[label_idx][0]).strip()
        digits = _collect_digits(texts, rects[rect_idx])
        grouped[label_text].extend(digits)

    result['entries'] = [
        {'label': lbl, 'numbers': sorted(nums)}
        for lbl, nums in sorted(grouped.items())
    ]
    result['unmatched_labels'] = sorted({
        normalize_width(tb_labels[i][0]).strip() for i in unmatched_idx
    })
    return result


def _format_numbers_with_counts(numbers):
    """整数リスト（重複可）を、出現回数が2回以上の値には `(件数)` を付けた
    昇順・カンマ区切りの表示文字列にする（例: [1, 2, 3, 7, 7] -> '1, 2, 3, 7(2)'）。"""
    counter = Counter(numbers)
    parts = []
    for n in sorted(counter):
        count = counter[n]
        parts.append(f'{n}({count})' if count > 1 else str(n))
    return ', '.join(parts)


_UNMATCHED_LABEL_MARKER = '端子検出不可'


def build_terminal_rows(results: dict) -> list:
    """{filename: analyze_dxf_terminals()結果} から、TB List シート用の
    行データ [{'端子台': str, '端子No.': str, '図番': str}, ...] を構築する。

    「端子台」（ラベルの正規化済みテキスト）でユニークをとる。複数ファイルに
    またがって同じ端子台名が登場する場合は、端子番号・図番を1行に統合する。
    「端子No.」は同じ番号が複数回登場する場合 `N(件数)` の形式で表示し
    （`_format_numbers_with_counts`）、図番は複数ある場合カンマ区切りで
    ABC順に列挙する。行自体も「端子台」のABC順に並べる。対象外ファイル
    （`is_target=False`）は集計対象に含めない。

    候補パターン（`^TB[A-Z0-9]+`）には一致したが対応する矩形が見つからな
    かったラベル（`unmatched_labels`）は、末尾に1行の空行を挟んだうえで、
    「端子台」列に `端子検出不可`、「端子No.」列にそのラベルテキスト、
    「図番」列にそのファイルの図番を記載する（ユーザー指定の仕様）。
    こちらも同じラベルが複数ファイルにまたがる場合は図番をABC順で統合し、
    行自体はラベルのABC順に並べる。
    """
    numbers_by_label = defaultdict(list)
    drawing_numbers_by_label = defaultdict(set)
    unmatched_drawing_numbers = defaultdict(set)

    for fname, data in results.items():
        if not data.get('is_target'):
            continue
        drawing_number = data.get('drawing_number') or os.path.splitext(fname)[0]
        for entry in data.get('entries', []):
            numbers_by_label[entry['label']].extend(entry['numbers'])
            drawing_numbers_by_label[entry['label']].add(drawing_number)
        for label in data.get('unmatched_labels', []):
            unmatched_drawing_numbers[label].add(drawing_number)

    rows = []
    for label in sorted(numbers_by_label.keys()):
        rows.append({
            '端子台': label,
            '端子No.': _format_numbers_with_counts(numbers_by_label[label]),
            '図番': ', '.join(sorted(drawing_numbers_by_label[label])),
        })

    if unmatched_drawing_numbers:
        rows.append({'端子台': '', '端子No.': '', '図番': ''})
        for label in sorted(unmatched_drawing_numbers.keys()):
            rows.append({
                '端子台': _UNMATCHED_LABEL_MARKER,
                '端子No.': label,
                '図番': ', '.join(sorted(unmatched_drawing_numbers[label])),
            })

    return rows
