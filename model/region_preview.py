"""
領域のブラウザ内プレビュー（モデル層。streamlit 非依存・DXF-extract-labels 専用）

元の図面全体を白一色で描いた SVG に、検出した領域の多角形（DXF 座標）を
色付きの輪郭とラベルで重ねた、自己完結の HTML（inline SVG + 素の JS）を作る。
DXF-viewer の Search Boundary（回路図に赤の境界）と同じ見せ方。

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
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ezdxf import recover
from ezdxf.addons.drawing import Frontend, RenderContext, config as drawing_config, layout, svg
from ezdxf.fonts import font_manager as ezdxf_font_manager
from ezdxf.fonts import fonts as ezdxf_fonts
from ezdxf.math import Matrix44, Vec3

from model import region_choices as rc

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
BASE_RENDER_VERSION = 2


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


def render_base_svg(dxf_bytes: bytes) -> BaseRender:
    """元の図面全体（モデル空間）を白一色の SVG にする。dxf_bytes は変更しない。"""
    doc, _auditor = recover.read(io.BytesIO(dxf_bytes))
    cjk_font = find_cjk_font()
    if cjk_font:
        for style in doc.styles:
            style.dxf.font = cjk_font
            if style.dxf.hasattr("bigfont"):
                style.dxf.bigfont = ""

    if len(doc.modelspace()) == 0:
        return BaseRender(cjk_font=cjk_font, empty=True)
    label_info = _collect_label_info(doc)

    cfg = drawing_config.Configuration(
        background_policy=drawing_config.BackgroundPolicy.BLACK,
        color_policy=drawing_config.ColorPolicy.CUSTOM,
        custom_fg_color=BASE_COLOR,
    )
    backend = svg.SVGBackend()
    Frontend(RenderContext(doc), backend, config=cfg).draw_layout(doc.modelspace())
    if not backend.player().bbox().has_data:
        return BaseRender(cjk_font=cjk_font, empty=True)

    root = backend.get_xml_root_element(layout.Page(0, 0, layout.Units.mm))
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
                            locked, match_fn, candidates_fn, unnamed=(), inactive=()):
    """viewer 部品に渡す data（JSON 化できる dict）を作る。streamlit 非依存。

    pending: 選択が必要な領域（reg dict のリスト）／decided: `[(reg, 確定した名称), ...]`
    state: 選択状態の正本（`st.session_state` 互換の dict。キー `rc_<fname>_<id>` のレコード。`model/region_choices.py`）
    unnamed: 名称が無い領域 `[(reg, 表示名, 採用するか), ...]`（採用する＝赤、しない＝灰。ラベルは元の図面に無いので多角形の左上）
    candidates_fn(reg, checked_names) -> [(距離, 文字列), ...]（共有側の `pending_candidates_for_region`）
    inactive: 抽出されない領域 `[(reg, 表示名, 注記), ...]`（「除外」「特定OFF」。灰色で表示し、名称は変更できない）
    match_fn(text, labels) -> 一致したラベル or None（共有側の `match_typed_region_name`）
    """
    labels = analysis.get("labels", [])

    def svg_pts(corners):
        return [_to_svg(base.matrix, x, y) for x, y in corners]

    def geoms(text):
        # 同じ文字列のラベルが複数あるときはすべて（領域が複数あるということ）
        return [dict(label_geom(base, lab), text=lab[0]) for lab in resolve_labels(text, labels)]

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
            "label": f"図面{r['frame'] + 1} / 領域{r['id'] + 1}（{r['area_pct']:.0f}%）",
            "points": [list(p) for p in pts], "anchor": list(min(pts, key=lambda p: (p[1], p[0]))),
            "geoms": geoms(shown_name) if shown_state == "decided" or note else [], "editable": True,
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
            "geoms": geoms(shown_name) if used and matched else [], "editable": True,
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
            "geoms": geoms(name), "editable": False,
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
                "geoms": geoms(t),
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
