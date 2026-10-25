"""
領域のブラウザ内プレビュー（モデル層。streamlit 非依存・DXF-extract-labels 専用）

元の図面全体を白一色で描いた SVG に、検出した領域の多角形（DXF 座標）を
色付きの輪郭で重ね、領域名・ラベルは**元の文字そのものの色を変えて**示す
自己完結の HTML（inline SVG + 素の JS）を作る。
DXF-viewer の Search Boundary（回路図に赤の境界）と同じ見せ方。

ラベルの色替えの仕組み（`render_base_svg()`）: ブロック内の文字だけをモデル空間へ移し、
文字1件ごとに一意の「目印の色」を付けて描く。ezdxf は色ごとに CSS クラスを作るので、
目印の色のクラスを控え（`BaseRender.label_classes`）、SVG の中身は白に書き換える。
viewer は、ハイライトするラベルのクラスの色だけを CSS で差し替える（描き直さない）。

DXF-diff-manager の model/dxf_preview.py（差分DXFのレイヤー選択ビューア）とは
表示の単位が違う（レイヤーではなく領域名）ため、別モジュールとして持つ。
同梱フォントの登録と pan/zoom の JS は、そちらから複製したもの（共有しない）。
viewer 部品の HTML/CSS/JS は `view/assets/region_confirm_viewer.*`（`view/region_confirm_viewer.py` が読む）。

設計上の要点:
- 領域の多角形はモデル空間座標。ezdxf が SVG 化したときの座標変換
  （get_xml_root_element の後の backend.transformation_matrix）で SVG 座標へ移す
- ezdxf が非表示（レイヤー OFF/凍結・invisible）を描かないので、見える図面と一致する
- 外部 JS・CDN には依存しない
"""

import io
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ezdxf import colors as ezdxf_colors
from ezdxf import recover
from ezdxf.addons.drawing import Frontend, RenderContext, config as drawing_config, layout, svg
from ezdxf.fonts import font_manager as ezdxf_font_manager
from ezdxf.fonts import fonts as ezdxf_fonts
from ezdxf.math import Matrix44, Vec3

from model import region_choices as rc
from model.common_utils import is_invisible
from model.ref_designator import OR_DELIMITER

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

# リポジトリ同梱フォント（IPAex ゴシック。無改変で再配布）。Streamlit Cloud には
# システムフォントが無いため、同梱して ezdxf のフォント管理に直接登録する
BUNDLED_FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"
BUNDLED_FONT_NAME = "ipaexg.ttf"
CJK_FONT_CANDIDATES = [
    BUNDLED_FONT_NAME, "ipag.ttf", "Arial Unicode.ttf", "ArialUni.ttf",
    "msgothic.ttc", "meiryo.ttc", "NotoSansCJK-Regular.ttc", "NotoSansCJKjp-Regular.otf",
]

BASE_COLOR = "#ffffff"  # 図面全体の色（2026-10-23 ユーザー指定: グレーは見づらいため白）

# 領域の見せ方: style -> (線色, 表示名)
STYLE_COLORS: Dict[str, Tuple[str, str]] = {
    "target": ("#ff3030", "特定／選択が必要"),
    "other": ("#ffa500", "未特定"),
    "excluded": ("#9a9a9a", "除外"),
}


# `BaseRender` の項目を変えたら上げる（app.py の cache_data のキーに含める）
BASE_RENDER_VERSION = 3


@dataclass
class BaseRender:
    inner_svg: str = ""
    view_box: Tuple[float, float] = (0.0, 0.0)
    matrix: Tuple[float, ...] = ()  # Matrix44 の 16 値（cache_data で保持できる形）
    cjk_font: Optional[str] = None
    empty: bool = False
    # 元の図面のラベルの配置: {(文字列, round(x,1), round(y,1)): (anchor, vpos, 文字の高さ, 回転角度)}
    # （`analysis['labels']` の座標と同じ。MTEXT/TEXT の挿入点は中央寄せ等の配置基準点のため、
    # 重ね書きする文字を元の位置に合わせるのに使う）
    labels: dict = field(default_factory=dict)
    # 元の文字ごとの CSS クラス: {(正規化したラベル, round(x,1), round(y,1)): [[クラス名, 'fill'|'stroke'], ...]}
    # （キーの文字列は `ref_designator.normalize_label()`、座標は `analysis['labels']` と同じ。viewer が
    # このクラスの色だけを差し替えて、元の文字のまま色を変える。同じキーに複数の実体があれば全部入る）
    label_classes: dict = field(default_factory=dict)


def _register_bundled_font() -> None:
    """同梱フォントを ezdxf のフォント管理に登録する（未登録のときだけ）。"""
    manager = ezdxf_fonts.font_manager
    if not (BUNDLED_FONT_DIR / BUNDLED_FONT_NAME).is_file():
        return
    if not manager.has_font(BUNDLED_FONT_NAME):
        manager.build([str(BUNDLED_FONT_DIR)], support_dirs=False)
    if BUNDLED_FONT_NAME not in ezdxf_font_manager.DEFAULT_FONTS:
        ezdxf_font_manager.DEFAULT_FONTS.append(BUNDLED_FONT_NAME)
    manager._fallback_font_name = ""


def find_cjk_font() -> Optional[str]:
    _register_bundled_font()
    manager = ezdxf_fonts.font_manager
    for name in CJK_FONT_CANDIDATES:
        if manager.has_font(name):
            return name
    return None


_MTEXT_ANCHOR = {1: ("start", "top"), 2: ("middle", "top"), 3: ("end", "top"),
                 4: ("start", "middle"), 5: ("middle", "middle"), 6: ("end", "middle"),
                 7: ("start", "bottom"), 8: ("middle", "bottom"), 9: ("end", "bottom")}
_TEXT_H = {0: "start", 1: "middle", 2: "end", 3: "start", 4: "middle", 5: "start"}
_TEXT_V = {0: "bottom", 1: "bottom", 2: "middle", 3: "top"}


def _collect_label_info(doc) -> dict:
    """`analyze_dxf_regions()` と同じ手順でラベル実体を集め、配置基準（anchor/vpos）・
    文字の高さ・回転角を `analysis['labels']` と同じキーで返す。失敗しても空 dict
    （その場合、文字は (x, y) を中央に置く）。"""
    try:
        import math
        from model.extract_labels import extract_text_from_entity
        from model.region_detector import DEFAULT_REGION_CONFIG, _collect_region_geometry
        entities = _collect_region_geometry(doc, dict(DEFAULT_REGION_CONFIG), True)[3]
    except Exception:
        return {}
    info = {}
    for e in entities:
        try:
            _raw, text, (x, y) = extract_text_from_entity(e)
            if not text:
                continue
            if e.dxftype() == 'MTEXT':
                anchor, vpos = _MTEXT_ANCHOR.get(e.dxf.get('attachment_point', 1), ("start", "top"))
                height = float(e.dxf.get('char_height', 0) or 0)
                td = e.dxf.get('text_direction', None)
                rot = math.degrees(math.atan2(td[1], td[0])) if td is not None else float(e.dxf.get('rotation', 0) or 0)
            else:
                anchor = _TEXT_H.get(e.dxf.get('halign', 0), "start")
                vpos = _TEXT_V.get(e.dxf.get('valign', 0), "bottom")
                height = float(e.dxf.get('height', 0) or 0)
                rot = float(e.dxf.get('rotation', 0) or 0)
            info.setdefault((text, round(x, 1), round(y, 1)), (anchor, vpos, height, rot))
        except Exception:
            continue
    return info


_TEXT_TYPES = ("TEXT", "MTEXT")
_MTEXT_COLOR_CODE = re.compile(r"\\[Cc][0-9]+;?")  # 本文中の色指定（\C1; \c255;）。目印の色に置き換える
_MAX_INSERT_DEPTH = 20
_STYLE_RULE = re.compile(r"^\.(C[0-9A-Fa-f]+)\s*\{(.*)\}\s*$", re.S)


def _insert_instances(insert):
    """INSERT 1件を描画上の実体に分ける（MINSERT は格子の要素ごと。`virtual_entities()` は先頭の要素しか返さない）。"""
    try:
        if (insert.dxf.get("row_count", 1) or 1) > 1 or (insert.dxf.get("column_count", 1) or 1) > 1:
            return list(insert.multi_insert())
    except Exception:
        pass
    return [insert]


def _lift_block_texts(doc) -> int:
    """ブロック内の TEXT/MTEXT を、モデル空間の実体に移す（個別に色を付けられるようにする）。

    線などの図形は INSERT のまま残す（INSERT をまるごと展開すると描画が遅く、線の座標もずれる）。
    移した文字は、たどった INSERT が参照するブロック定義（入れ子を含む）からだけ取り除く。
    寸法線（DIMENSION）の寸法値は INSERT から参照されない無名ブロック（`*D…`）の中の文字なので、
    全ブロックから取り除くと寸法値が消える。非表示（invisible 属性・画層 OFF）の文字は移さない
    （ezdxf も描かないので、見える図面と一致する）。画層 0 の文字は、親 INSERT の画層で描かれるため、
    その画層に付け替える。戻り値は移した文字の数。"""
    msp = doc.modelspace()
    traversed = set()

    def collect(insert, parent_layer, depth):
        out = []
        if depth > _MAX_INSERT_DEPTH:
            return out
        traversed.add(insert.dxf.name)
        try:
            for v in insert.virtual_entities():
                if is_invisible(v):
                    continue
                layer = v.dxf.layer if v.dxf.layer != "0" else parent_layer
                t = v.dxftype()
                if t in _TEXT_TYPES:
                    v.dxf.layer = layer
                    out.append(v)
                elif t == "INSERT":
                    for inst in _insert_instances(v):
                        out.extend(collect(inst, layer, depth + 1))
        except Exception:
            pass
        return out

    lifted = []
    for e in list(msp.query("INSERT")):
        if is_invisible(e):
            continue
        for inst in _insert_instances(e):
            lifted.extend(collect(inst, e.dxf.layer, 0))
    for blk in doc.blocks:
        if blk.is_any_layout or blk.name not in traversed:
            continue
        for x in list(blk):
            if x.dxftype() in _TEXT_TYPES:
                blk.delete_entity(x)
    for v in lifted:
        msp.add_entity(v)
    return len(lifted)


def _assign_marker_colors(doc) -> Dict[str, tuple]:
    """モデル空間の TEXT/MTEXT すべてに、1件ごとに一意の「目印の色」を付ける。

    戻り値: {目印の色（'#rrggbb'）: ラベルのキー (正規化した文字列, round(x,1), round(y,1)) または None}。
    赤成分を 0x80 以上にするのは、暗い色を ezdxf が背景に合わせて置き換えないようにするため。"""
    from model.extract_labels import extract_text_from_entity
    from model.ref_designator import normalize_label
    marker_to_key: Dict[str, tuple] = {}
    n = 0
    for e in doc.modelspace().query("TEXT MTEXT"):
        n += 1
        rgb = (0x80 | ((n >> 16) & 0x7F), (n >> 8) & 0xFF, n & 0xFF)
        key = None
        try:
            _raw, text, (x, y) = extract_text_from_entity(e)
            text = normalize_label(text) if text else ""
            if text:
                key = (text, round(x, 1), round(y, 1))
        except Exception:
            pass
        e.dxf.true_color = ezdxf_colors.rgb2int(rgb)
        if e.dxftype() == "MTEXT":
            # 本文中の色指定は実体の色を打ち消す。取り除くと「書式付き」から「単純な」MTEXT の描き方に変わって
            # 文字の位置が僅かにずれるので、目印の色（`\c` は BGR の整数）に置き換えて書式付きのままにする
            bgr = (rgb[2] << 16) | (rgb[1] << 8) | rgb[0]
            e.text = _MTEXT_COLOR_CODE.sub(lambda _m: "\\c%d;" % bgr, e.text)
        marker_to_key["#%02x%02x%02x" % rgb] = key
    return marker_to_key


def _rewrite_marker_styles(root, marker_to_key) -> dict:
    """SVG の `<style>` のうち、目印の色（fill/stroke）を使うクラスを白（`BASE_COLOR`）に書き換え、
    `label_classes`（ラベルのキー -> [[クラス名, 'fill'|'stroke'], ...]）を返す。"""
    label_classes: dict = {}
    for el in root.iter():
        if el.tag.split("}")[-1] != "style" or not el.text:
            continue
        m = _STYLE_RULE.match(el.text.strip())
        if not m:
            continue
        cls, body = m.group(1), m.group(2)
        for kind in ("fill", "stroke"):
            km = re.search(r"%s:\s*(#[0-9a-fA-F]{6})\b" % kind, body)
            if not km or km.group(1).lower() not in marker_to_key:
                continue
            key = marker_to_key[km.group(1).lower()]
            body = body.replace(km.group(0), "%s: %s" % (kind, BASE_COLOR))
            if key is not None:
                label_classes.setdefault(key, []).append([cls, kind])
        el.text = ".%s {%s}" % (cls, body)
    return label_classes


def render_base_svg(dxf_bytes: bytes) -> BaseRender:
    """元の図面全体（モデル空間）を白一色の SVG にする。dxf_bytes は変更しない。

    見た目は白一色だが、文字（TEXT/MTEXT）はラベルごとに専用の CSS クラスを持つ
    （`BaseRender.label_classes`。仕組みはモジュール先頭の docstring）。"""
    doc, _auditor = recover.read(io.BytesIO(dxf_bytes))
    cjk_font = find_cjk_font()
    if cjk_font:
        for style in doc.styles:
            style.dxf.font = cjk_font
            if style.dxf.hasattr("bigfont"):
                style.dxf.bigfont = ""

    if len(doc.modelspace()) == 0:
        return BaseRender(cjk_font=cjk_font, empty=True)
    label_info = _collect_label_info(doc)  # 文字を移す前の doc で集める（抽出と同じ手順）

    _lift_block_texts(doc)
    marker_to_key = _assign_marker_colors(doc)

    def keep_base_color(entity, properties):
        # 文字の色は実体に付けた目印の色（書式付き MTEXT はこのフックでは変えられない）。文字以外は白
        if entity.dxftype() not in _TEXT_TYPES:
            properties.color = BASE_COLOR

    cfg = drawing_config.Configuration(
        background_policy=drawing_config.BackgroundPolicy.BLACK,
        color_policy=drawing_config.ColorPolicy.COLOR,
    )
    backend = svg.SVGBackend()
    frontend = Frontend(RenderContext(doc), backend, config=cfg)
    frontend.push_property_override_function(keep_base_color)
    frontend.draw_layout(doc.modelspace())
    if not backend.player().bbox().has_data:
        return BaseRender(cjk_font=cjk_font, empty=True)

    root = backend.get_xml_root_element(layout.Page(0, 0, layout.Units.mm))
    label_classes = _rewrite_marker_styles(root, marker_to_key)
    vb = [float(v) for v in root.get("viewBox", "0 0 0 0").split()]
    parts: List[str] = []
    for child in root:
        tag = child.tag.split("}")[-1]
        if tag in ("defs", "g"):
            parts.append(ET.tostring(child, encoding="unicode"))
    return BaseRender(
        inner_svg="".join(parts),
        view_box=(vb[2], vb[3]),
        matrix=tuple(float(v) for v in backend.transformation_matrix),
        cjk_font=cjk_font,
        labels=label_info,
        label_classes=label_classes,
    )


def _to_svg(matrix: Tuple[float, ...], x: float, y: float) -> Tuple[float, float]:
    v = Matrix44(matrix).transform(Vec3(x, y, 0))
    return float(v.x), float(v.y)


def resolve_labels(text, labels):
    """`labels`（`(text, x, y)`）のうち、`text` と NFKC・大文字で一致する元ラベルをすべて返す。

    同じ文字列が図面に複数あるのは領域が複数あるということなので、1件に絞らず
    すべてハイライトする（2026-10-23 ユーザー決定）。"""
    key = _norm(text)
    return [lab for lab in labels if _norm(lab[0]) == key]


def label_geom(base: BaseRender, label) -> dict:
    """元ラベル `(text, x, y)` を、SVG 上で元の位置・大きさ・向きに重ねるための配置情報にする。

    {x, y: SVG 座標, anchor: 'start'|'middle'|'end', base: 'hanging'|'central'|'text-after-edge',
     size: 文字の高さ(SVG単位。不明なら None), rot: SVG での回転角度}"""
    text, x, y = label
    anchor, vpos, height, rot = getattr(base, "labels", {}).get((text, round(x, 1), round(y, 1)),
                                                ("middle", "middle", 0.0, 0.0))
    sx, sy = _to_svg(base.matrix, x, y)
    ux, uy = _to_svg(base.matrix, x + 1.0, y)
    scale = ((ux - sx) ** 2 + (uy - sy) ** 2) ** 0.5
    return {
        "x": sx, "y": sy, "anchor": anchor,
        "base": {"top": "hanging", "middle": "central", "bottom": "text-after-edge"}.get(vpos, "central"),
        # DXF の文字の高さは大文字の高さ。SVG の font-size（全体の高さ）は約 1.4 倍にすると元の見た目に近い
        "size": (height * scale * 1.4) if height and scale else None,
        "rot": -rot,  # SVG は y 下向き
    }


def label_classes_of(base: BaseRender, label) -> list:
    """元ラベル `(text, x, y)` の文字に付いた CSS クラス `[[クラス名, 'fill'|'stroke'], ...]`。
    図面に無い（`label_classes` に無い）ラベルは空リスト（viewer は従来の重ね書きにする）。"""
    from model.ref_designator import normalize_label
    text, x, y = label
    return [list(c) for c in getattr(base, "label_classes", {}).get(
        (normalize_label(text), round(x, 1), round(y, 1)), [])]


def region_label_occurrences(text, labels, polygon):
    """領域名 `text` として強調する元ラベル `(text, x, y)` を返す。

    `text` と NFKC・大文字で一致するラベルのうち、領域の多角形の内側（境界線上を含む）にあるものすべて。
    内側に1件も無ければ、多角形に最も近い1件。領域名の候補は領域の内側のラベルから作られるので、
    領域から離れた同名のラベル（図面情報枠など）は強調しない。入力名（領域外のラベルもあり得る）や
    親の候補を引き継ぐ小領域は、内側に無いので最も近い1件になる。"""
    from model.region_detector import _point_in_polygon
    hits = resolve_labels(text, labels)
    if not hits or not polygon:
        return hits
    inside = [lab for lab in hits if _point_in_polygon((lab[1], lab[2]), polygon)]
    if inside:
        return inside
    return [min(hits, key=lambda lab: _dist_point_polygon(lab[1], lab[2], polygon))]


COLOR_PENDING = "#ff9f1a"
COLOR_CAND_SELECTED = "#ff9f1a"
COLOR_CAND_OTHER = "#ffe600"
COLOR_DECIDED = "#ff3030"
COLOR_OFF = "#9a9a9a"  # 名称が無く「特定」OFF（採用しない）の領域


def _norm(text):
    import unicodedata
    return unicodedata.normalize("NFKC", text or "").strip().upper()


def _dist_point_polygon(px, py, polygon) -> float:
    """点から多角形の辺までの最短距離（内側でも辺までの距離。境界に近いラベルを選ぶため）。"""
    best = float("inf")
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        L = dx * dx + dy * dy
        t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L))
        d = ((px - (x1 + t * dx)) ** 2 + (py - (y1 + t * dy)) ** 2) ** 0.5
        best = min(best, d)
    return best


def _poly_of(reg):
    return [tuple(c) for c in (reg.get("corners") or reg.get("polygon") or [])]


def build_confirmation_data(base, fname, analysis, pending, decided, state, checked_names,
                            locked, match_fn, candidates_fn, unnamed=(), inactive=(), minor_ids=()):
    """viewer 部品に渡す data（JSON 化できる dict）を作る。streamlit 非依存。

    pending: 選択が必要な領域（reg dict のリスト）／decided: `[(reg, 確定した名称), ...]`
    state: 選択状態の正本（`st.session_state` 互換の dict。キー `rc_<fname>_<id>` のレコード。`model/region_choices.py`）
    unnamed: 名称が無い領域 `[(reg, 表示名, 採用するか), ...]`（採用する＝赤、しない＝灰。ラベルは元の図面に無いので多角形の左上）
    candidates_fn(reg, checked_names) -> [(距離, 文字列), ...]（共有側の `pending_candidates_for_region`）
    inactive: 抽出されない領域 `[(reg, 表示名, 注記), ...]`（「除外」「特定OFF」。灰色で表示し、名称は変更できない）
    match_fn(text, labels) -> 一致したラベル or None（共有側の `match_typed_region_name`）
    minor_ids: `decided` のうち、5%未満などの小領域（`minor_regions`）の id（表示ラベルに「小領域」を付ける）
    """
    labels = analysis.get("labels", [])

    def svg_pts(corners):
        return [_to_svg(base.matrix, x, y) for x, y in corners]

    def geoms(text, reg):
        # その領域の内側の同名ラベルすべて（なければ最も近い1件）。`cls` は元の文字の CSS クラス
        # 名称が ` | `（どれか1つの意味。連結された名称）のときは、各名称のラベルを強調する
        return [dict(label_geom(base, lab), text=lab[0], cls=label_classes_of(base, lab))
                for part in text.split(OR_DELIMITER)
                for lab in region_label_occurrences(part, labels, _poly_of(reg))]

    def area_label(r):
        suffix = "・小領域" if r['id'] in minor_ids else ""
        return f"図面{r['frame'] + 1} / 領域{r['id'] + 1}（{r['area_pct']:.0f}%{suffix}）"

    regions, choices = [], {}

    def override(r):
        """確定済み・無名の領域に「領域名を入力」を使っているときの (使用中か, 一致したラベル or None)。"""
        if not rc.get_value(state, fname, r['id'], 'typed', False):
            return False, None
        return True, match_fn(rc.get_value(state, fname, r['id'], 'typedtext', "") or "", labels)

    def editable_choice(r):
        typedtext = rc.get_value(state, fname, r['id'], 'typedtext', "") or ""
        used, matched = override(r)
        return {"cands": [], "none": bool(rc.get_value(state, fname, r['id'], 'skip', False)),
                "typed": used, "typedtext": typedtext,
                "typedMatch": matched if used else None, "editable": True}

    def is_none(r):
        return bool(rc.get_value(state, fname, r['id'], 'skip', False))

    for r, name in decided:
        corners = _poly_of(r)
        if not corners:
            continue
        pts = svg_pts(corners)
        used, matched = override(r)
        shown_state, shown_name = "decided", name
        note = ""
        if is_none(r):  # 「領域選択しない」は入力名より優先（入力名は保持され、外すと戻る）
            shown_state, note = "unnamed_off", "領域選択しない"
        elif used:  # 入力した領域名が優先（一致しなければ未選択）
            shown_state, shown_name = ("decided", matched) if matched else ("unnamed_off", "（入力名が一致せず未選択）")
        regions.append({
            "key": f"d{r['id']}", "id": r["id"], "state": shown_state, "name": shown_name, "note": note,
            "label": area_label(r),
            "points": [list(p) for p in pts], "anchor": list(min(pts, key=lambda p: (p[1], p[0]))),
            "geoms": geoms(shown_name, r) if shown_state == "decided" or note else [], "editable": True,
        })
        choices[str(r["id"])] = editable_choice(r)
    for r, label, adopted in unnamed:
        corners = _poly_of(r)
        if not corners:
            continue
        pts = svg_pts(corners)
        used, matched = override(r)
        shown_state, shown_name = ("unnamed" if adopted else "unnamed_off"), label
        note = "" if adopted else "特定OFF"
        if is_none(r):
            shown_state, note = "unnamed_off", "領域選択しない"
        elif used:
            shown_state, shown_name = ("decided", matched) if matched else ("unnamed_off", "（入力名が一致せず未選択）")
            note = ""
        regions.append({
            "key": f"u{r['id']}", "id": r["id"], "state": shown_state, "name": shown_name,
            "label": f"図面{r['frame'] + 1} / 領域{r['id'] + 1}（{r['area_pct']:.0f}%）",
            "points": [list(p) for p in pts], "anchor": list(min(pts, key=lambda p: (p[1], p[0]))),
            "geoms": geoms(shown_name, r) if used and matched else [], "editable": True,
            "note": note,
        })
        choices[str(r["id"])] = editable_choice(r)
    for r, name, note in inactive:
        corners = _poly_of(r)
        if not corners:
            continue
        pts = svg_pts(corners)
        regions.append({
            "key": f"x{r['id']}", "id": r["id"], "state": "unnamed_off", "name": name, "note": note,
            "label": f"図面{r['frame'] + 1} / 領域{r['id'] + 1}（{r['area_pct']:.0f}%）",
            "points": [list(p) for p in pts], "anchor": list(min(pts, key=lambda p: (p[1], p[0]))),
            "geoms": geoms(name, r), "editable": False,
        })
    for r in pending:
        corners = _poly_of(r)
        if not corners:
            continue
        rid = r["id"]
        val = lambda suffix, default=None: rc.get_value(state, fname, rid, suffix, default)
        pts = svg_pts(corners)
        cands = []
        for i, (d, t) in enumerate(candidates_fn(r, checked_names)):
            cands.append({
                "idx": i, "text": t, "dist": round(float(d), 1),
                "selected": bool(val(i, False)),
                "geoms": geoms(t, r),
            })
        typedtext = val("typedtext", "") or ""
        regions.append({
            "key": f"p{rid}", "id": rid, "state": "pending", "name": "",
            "label": f"図面{r['frame'] + 1} / 領域{rid + 1}（{r['area_pct']:.0f}%）",
            "points": [list(p) for p in pts], "anchor": list(min(pts, key=lambda p: (p[1], p[0]))),
        })
        choices[str(rid)] = {
            "cands": cands, "none": bool(val("none", False)),
            "typed": bool(val("typed", False)), "typedtext": typedtext,
            "typedMatch": match_fn(typedtext, labels) if typedtext else None,
        }
    return {
        "fname": fname, "viewBox": [base.view_box[0], base.view_box[1]],
        "svg": base.inner_svg, "sig": f"{fname}:{len(base.inner_svg)}:{hash(base.inner_svg) & 0xffffffff}",
        "regions": regions, "choices": choices, "locked": bool(locked),
        "colors": {"pending": COLOR_PENDING, "candSel": COLOR_CAND_SELECTED,
                   "candOther": COLOR_CAND_OTHER, "decided": COLOR_DECIDED, "off": COLOR_OFF},
        "hasFont": base.cjk_font is not None, "empty": base.empty,
        "labelNames": sorted({lab[0] for lab in labels if lab[0]}),
    }
