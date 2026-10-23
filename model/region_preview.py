"""
領域のブラウザ内プレビュー（モデル層。streamlit 非依存・DXF-extract-labels 専用）

元の図面全体を白一色で描いた SVG に、検出した領域の多角形（DXF 座標）を
色付きの輪郭とラベルで重ねた、自己完結の HTML（inline SVG + 素の JS）を作る。
DXF-viewer の Search Boundary（回路図に赤の境界）と同じ見せ方。

DXF-diff-manager の model/dxf_preview.py（差分DXFのレイヤー選択ビューア）とは
表示の単位が違う（レイヤーではなく領域名）ため、別モジュールとして持つ。
同梱フォントの登録と pan/zoom の JS は、そちらから複製したもの（共有しない）。

設計上の要点:
- 領域の多角形はモデル空間座標。ezdxf が SVG 化したときの座標変換
  （get_xml_root_element の後の backend.transformation_matrix）で SVG 座標へ移す
- ezdxf が非表示（レイヤー OFF/凍結・invisible）を描かないので、見える図面と一致する
- 外部 JS・CDN には依存しない
"""

import html
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


@dataclass
class RegionOverlay:
    key: str
    label: str
    corners: List[Tuple[float, float]]  # DXF（モデル空間）座標
    style: str = "target"  # 'target' | 'other' | 'excluded'
    checked: bool = True
    label_xys: List[Tuple[str, float, float]] = field(default_factory=list)  # 元の図面の該当ラベル (text, x, y) すべて。無ければ多角形の左上


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
    anchor, vpos, height, rot = base.labels.get((text, round(x, 1), round(y, 1)),
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


def label_text_svg(geom: dict, text: str, color: str) -> str:
    """`label_geom()` の配置で色付き文字を作る（黒の縁取りで下の元の文字を隠す）。"""
    size = geom.get("size")
    style = f"font-size:{size:.2f}px;stroke-width:{size * 0.25:.2f}px" if size else ""
    rot = f' transform="rotate({geom["rot"]:.2f} {geom["x"]:.2f} {geom["y"]:.2f})"' if geom.get("rot") else ""
    return (
        f'<text class="lb" x="{geom["x"]:.2f}" y="{geom["y"]:.2f}" fill="{color}" '
        f'text-anchor="{geom["anchor"]}" dominant-baseline="{geom["base"]}"{rot} '
        f'style="{style}">{html.escape(text)}</text>'
    )


def confirmation_overlays(pending_regions) -> List[RegionOverlay]:
    """「領域の確認」で選択が必要な領域をすべて赤（target）・表示ONにする。"""
    return [
        RegionOverlay(
            key=f"r{reg['id']}",
            label=f"図面{reg['frame'] + 1} / 領域{reg['id'] + 1}（{reg['area_pct']:.0f}%）",
            corners=[tuple(c) for c in reg.get('corners') or reg.get('polygon') or []],
            style="target",
            checked=True,
        )
        for reg in pending_regions
    ]


_MARK_STYLE = {"特定": ("target", True), "": ("other", False), "除外": ("excluded", False)}


def preview_overlays(analysis, region_rows) -> List[RegionOverlay]:
    """抽出結果の領域行（region_rows）から overlay を作る。

    特定→赤・ON／未特定→橙・OFF／除外→灰・OFF。多角形は analysis の
    regions と minor_regions から id で引く（見つからない行は飛ばす）。
    """
    polys = {}
    for reg in list(analysis.get('regions', [])) + list(analysis.get('minor_regions', [])):
        polys[reg['id']] = reg
    overlays: List[RegionOverlay] = []
    for row in region_rows:
        reg = polys.get(row.get('id'))
        if reg is None:
            continue
        corners = [tuple(c) for c in reg.get('corners') or reg.get('polygon') or []]
        if not corners:
            continue
        style, checked = _MARK_STYLE.get(row.get('mark', ''), ("other", False))
        name = row.get('name') or f"領域{row['id'] + 1}"
        label_xys = []
        for part in [p.strip() for p in name.replace(' | ', ',').split(',') if p.strip()]:
            label_xys = resolve_labels(part, analysis.get('labels', []))
            if label_xys:
                break
        overlays.append(RegionOverlay(
            key=f"r{row['id']}",
            label=f"{name}（{row.get('area_pct', 0):.0f}%）",
            corners=corners,
            style=style,
            checked=checked,
            label_xys=label_xys,
        ))
    return overlays


_VIEWER_CSS = """
html,body{margin:0;height:100%;background:#000;color:#ddd;
  font-family:-apple-system,"Hiragino Sans","Yu Gothic UI","Meiryo",sans-serif;font-size:13px}
#wrap{display:flex;height:100%}
#side{width:260px;flex:none;padding:8px;box-sizing:border-box;overflow:auto;
  background:#1c1c1c;border-right:1px solid #333}
#side h4{margin:6px 0 4px;font-size:12px;color:#aaa;font-weight:600}
#side label{display:flex;align-items:center;gap:6px;margin:3px 0;cursor:pointer;line-height:1.3}
.sw{display:inline-block;width:12px;height:12px;border:1px solid #666;flex:none}
button{background:#2d2d2d;color:#ddd;border:1px solid #555;border-radius:4px;
  padding:3px 8px;margin:2px 2px 2px 0;cursor:pointer;font-size:12px}
button:hover{background:#3a3a3a}
#stage{flex:1;min-width:0;position:relative}
#cv{width:100%;height:100%;display:block;cursor:grab;touch-action:none}
#cv.drag{cursor:grabbing}
#cv path{vector-effect:non-scaling-stroke;stroke-width:1px}
#cv .ov polygon{vector-effect:non-scaling-stroke;stroke-width:2px;fill-opacity:.10}
#cv .ov text{font-size:var(--fs,14px);paint-order:stroke;stroke:#000;stroke-width:calc(var(--fs,14px)*.25);
  stroke-linejoin:round;pointer-events:none}
#note{position:absolute;right:8px;bottom:6px;color:#777;font-size:11px;pointer-events:none}
"""

# pan/zoom は model/dxf_preview.py（DXF-diff-manager 正本）の JS を複製
_VIEWER_JS = """
(function(){
  var svg=document.getElementById('cv');
  var vb0=svg.getAttribute('viewBox').split(' ').map(Number);
  var vb=vb0.slice();
  function apply(){svg.setAttribute('viewBox',vb.join(' '));
    var r=svg.getBoundingClientRect();
    if(r.width&&r.height)svg.style.setProperty('--fs',(Math.max(vb[2]/r.width,vb[3]/r.height)*13)+'px');}
  function reset(){vb=vb0.slice();apply();}
  function pt(e){var r=svg.getBoundingClientRect();
    var s=Math.max(vb[2]/r.width,vb[3]/r.height);
    var offx=(r.width-vb[2]/s)/2, offy=(r.height-vb[3]/s)/2;
    return {x:vb[0]+(e.clientX-r.left-offx)*s,y:vb[1]+(e.clientY-r.top-offy)*s,s:s};}
  svg.addEventListener('wheel',function(e){
    e.preventDefault();
    var p=pt(e),k=e.deltaY<0?0.8:1.25;
    var nw=vb[2]*k,nh=vb[3]*k;
    if(nw>vb0[2]*20||nw<vb0[2]/2000)return;
    vb=[p.x-(p.x-vb[0])*k,p.y-(p.y-vb[1])*k,nw,nh];apply();
  },{passive:false});
  var drag=null;
  svg.addEventListener('pointerdown',function(e){
    drag={x:e.clientX,y:e.clientY,vb:vb.slice(),s:pt(e).s};
    svg.setPointerCapture(e.pointerId);svg.classList.add('drag');});
  svg.addEventListener('pointermove',function(e){
    if(!drag)return;
    vb=[drag.vb[0]-(e.clientX-drag.x)*drag.s,drag.vb[1]-(e.clientY-drag.y)*drag.s,drag.vb[2],drag.vb[3]];apply();});
  function end(){drag=null;svg.classList.remove('drag');}
  svg.addEventListener('pointerup',end);svg.addEventListener('pointercancel',end);
  svg.addEventListener('dblclick',reset);
  document.getElementById('fit').addEventListener('click',reset);
  apply();
  var wrap=document.getElementById('wrap'),fs=document.getElementById('fs');
  fs.addEventListener('click',function(){
    var note=document.getElementById('fsnote');note.textContent='';
    if(document.fullscreenElement){document.exitFullscreen();return;}
    var p=wrap.requestFullscreen?wrap.requestFullscreen():null;
    if(!p){note.textContent='この環境では全画面にできません';return;}
    p.catch(function(){note.textContent='この環境では全画面にできません';});});
  document.addEventListener('fullscreenchange',function(){
    fs.textContent=document.fullscreenElement?'全画面を終了':'全画面';
    wrap.style.height=document.fullscreenElement?'100vh':wrap.getAttribute('data-h');});

  var boxes=Array.prototype.slice.call(document.querySelectorAll('input[data-ov]'));
  function show(key,on){
    var g=svg.querySelector('g.ov[data-ov="'+key+'"]');
    if(g)g.style.display=on?'':'none';}
  boxes.forEach(function(b){
    show(b.getAttribute('data-ov'),b.checked);
    b.addEventListener('change',function(){show(b.getAttribute('data-ov'),b.checked);});});
  document.querySelectorAll('button[data-preset]').forEach(function(btn){
    btn.addEventListener('click',function(){
      var want=btn.getAttribute('data-preset');
      boxes.forEach(function(b){
        var on=want==='all'||(want==='target'&&b.getAttribute('data-style')==='target');
        b.checked=on;show(b.getAttribute('data-ov'),on);});});});
})();
"""


def _overlay_svg(base: BaseRender, ov: RegionOverlay) -> str:
    color = STYLE_COLORS.get(ov.style, STYLE_COLORS["other"])[0]
    pts = [_to_svg(base.matrix, x, y) for x, y in ov.corners]
    if not pts:
        return ""
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
    dash = ' stroke-dasharray="6 4"' if ov.style == "excluded" else ""
    if ov.label_xys:
        # 元の図面の該当ラベルすべてに、元の位置・大きさで色付き文字を重ねる
        name = ov.label.rsplit("（", 1)[0]
        texts = "".join(label_text_svg(label_geom(base, lab), lab[0], color) for lab in ov.label_xys)
    else:
        # 該当ラベルが無い（no name 等）: 多角形の左上に表示する
        tx, ty = min(pts, key=lambda p: (p[1], p[0]))
        texts = f'<text x="{tx + 4:.2f}" y="{ty + 16:.2f}" fill="{color}">{html.escape(ov.label)}</text>'
    return (
        f'<g class="ov" data-ov="{html.escape(ov.key)}">'
        f'<polygon points="{points}" fill="{color}" stroke="{color}"{dash}/>{texts}</g>'
    )


def build_region_viewer_html(base: BaseRender, overlays: List[RegionOverlay], height_px: int = 720) -> str:
    """自己完結した HTML（inline SVG + CSS + 素の JS）を生成する。外部参照は含まない。"""
    if base.empty:
        return (
            f'<div style="height:{height_px}px;display:flex;align-items:center;'
            'justify-content:center;background:#000;color:#aaa;font-family:sans-serif">'
            "描画対象のエンティティがありません</div>"
        )
    w, h = base.view_box
    rows = []
    for ov in overlays:
        color = STYLE_COLORS.get(ov.style, STYLE_COLORS["other"])[0]
        rows.append(
            f'<label><input type="checkbox" data-ov="{html.escape(ov.key)}" '
            f'data-style="{html.escape(ov.style)}" {"checked" if ov.checked else ""}>'
            f'<span class="sw" style="background:{color}"></span>'
            f"<span>{html.escape(ov.label)}</span></label>"
        )
    if not rows:
        rows.append('<div style="color:#888">表示できる領域がありません</div>')
    legend = "".join(
        f'<div><span class="sw" style="background:{c}"></span> {html.escape(n)}</div>'
        for c, n in STYLE_COLORS.values()
    )
    body = base.inner_svg + "".join(_overlay_svg(base, ov) for ov in overlays)
    font_note = "" if base.cjk_font else "日本語フォントが見つからないため、日本語の文字は表示されません。"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>{_VIEWER_CSS}</style></head>
<body><div id="wrap" data-h="{height_px}px" style="height:{height_px}px">
<div id="side">
<h4>表示プリセット</h4>
<button data-preset="target">特定のみ</button><button data-preset="all">すべて</button><button data-preset="none">なし</button>
<h4>領域</h4>{"".join(rows)}
<h4>凡例</h4><div style="line-height:1.6;color:#aaa">{legend}</div>
<h4>操作</h4><button id="fit">全体表示</button><button id="fs">全画面</button><div id="fsnote" style="color:#e8a"></div>
<div style="color:#888;margin-top:6px;line-height:1.5">ホイール: 拡大縮小<br>ドラッグ: 移動<br>ダブルクリック: 全体表示</div>
<div style="color:#e8a;margin-top:8px">{html.escape(font_note)}</div>
</div>
<div id="stage"><svg id="cv" xmlns="{SVG_NS}" viewBox="0 0 {w:g} {h:g}" preserveAspectRatio="xMidYMid meet">
<rect x="0" y="0" width="{w:g}" height="{h:g}" fill="#000"/>{body}</svg>
<div id="note">線幅は拡大しても一定です</div></div>
</div><script>{_VIEWER_JS}</script></body></html>"""


# ============================================================
# 「領域の確認」の双方向 viewer（st.components.v2 の部品。Streamlit 側で登録する）
# ============================================================
# 色: 選択が必要な領域＝橙／選択中の候補ラベル＝橙／それ以外の候補ラベル＝黄／確定した領域＝赤
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
                            locked, match_fn, candidates_fn, unnamed=()):
    """viewer 部品に渡す data（JSON 化できる dict）を作る。streamlit 非依存。

    pending: 選択が必要な領域（reg dict のリスト）／decided: `[(reg, 確定した名称), ...]`
    state: 選択状態の正本（`st.session_state` 互換の dict。キー `rc_<fname>_<id>_…`）
    unnamed: 名称が無い領域 `[(reg, 表示名, 採用するか), ...]`（採用する＝赤、しない＝灰。ラベルは元の図面に無いので多角形の左上）
    candidates_fn(reg, checked_names) -> [(距離, 文字列), ...]（共有側の `pending_candidates_for_region`）
    match_fn(text, labels) -> 一致したラベル or None（共有側の `match_typed_region_name`）
    """
    labels = analysis.get("labels", [])

    def svg_pts(corners):
        return [_to_svg(base.matrix, x, y) for x, y in corners]

    def geoms(text):
        # 同じ文字列のラベルが複数あるときはすべて（領域が複数あるということ）
        return [dict(label_geom(base, lab), text=lab[0]) for lab in resolve_labels(text, labels)]

    regions, choices = [], {}
    for reg in decided:
        r, name = reg
        corners = _poly_of(r)
        if not corners:
            continue
        pts = svg_pts(corners)
        regions.append({
            "key": f"d{r['id']}", "id": r["id"], "state": "decided", "name": name,
            "label": f"図面{r['frame'] + 1} / 領域{r['id'] + 1}（{r['area_pct']:.0f}%）",
            "points": [list(p) for p in pts], "anchor": list(min(pts, key=lambda p: (p[1], p[0]))),
            "geoms": geoms(name),
        })
    for r, label, adopted in unnamed:
        corners = _poly_of(r)
        if not corners:
            continue
        pts = svg_pts(corners)
        regions.append({
            "key": f"u{r['id']}", "id": r["id"], "state": "unnamed" if adopted else "unnamed_off", "name": label,
            "label": f"図面{r['frame'] + 1} / 領域{r['id'] + 1}（{r['area_pct']:.0f}%）",
            "points": [list(p) for p in pts], "anchor": list(min(pts, key=lambda p: (p[1], p[0]))),
            "geoms": [],
        })
    for r in pending:
        corners = _poly_of(r)
        if not corners:
            continue
        rid = r["id"]
        k = lambda suffix: f"rc_{fname}_{rid}_{suffix}"
        pts = svg_pts(corners)
        cands = []
        for i, (d, t) in enumerate(candidates_fn(r, checked_names)):
            cands.append({
                "idx": i, "text": t, "dist": round(float(d), 1),
                "selected": bool(state.get(k(i), False)),
                "geoms": geoms(t),
            })
        typedtext = state.get(k("typedtext"), "") or ""
        regions.append({
            "key": f"p{rid}", "id": rid, "state": "pending", "name": "",
            "label": f"図面{r['frame'] + 1} / 領域{rid + 1}（{r['area_pct']:.0f}%）",
            "points": [list(p) for p in pts], "anchor": list(min(pts, key=lambda p: (p[1], p[0]))),
        })
        choices[str(rid)] = {
            "cands": cands, "none": bool(state.get(k("none"), False)),
            "typed": bool(state.get(k("typed"), False)), "typedtext": typedtext,
            "typedMatch": match_fn(typedtext, labels) if typedtext else None,
        }
    return {
        "fname": fname, "viewBox": [base.view_box[0], base.view_box[1]],
        "svg": base.inner_svg, "sig": f"{fname}:{len(base.inner_svg)}:{hash(base.inner_svg) & 0xffffffff}",
        "regions": regions, "choices": choices, "locked": bool(locked),
        "colors": {"pending": COLOR_PENDING, "candSel": COLOR_CAND_SELECTED,
                   "candOther": COLOR_CAND_OTHER, "decided": COLOR_DECIDED, "off": COLOR_OFF},
        "hasFont": base.cjk_font is not None, "empty": base.empty,
    }


CONFIRM_HTML = '<div class="rp-wrap" id="rp-wrap"></div>'

CONFIRM_CSS = """
.rp-wrap{display:flex;height:100%;min-height:300px;background:#000;color:#ddd;
  font-family:-apple-system,"Hiragino Sans","Yu Gothic UI","Meiryo",sans-serif;font-size:13px}
.rp-side{width:300px;flex:none;padding:8px;box-sizing:border-box;overflow:auto;
  background:#1c1c1c;border-right:1px solid #333}
.rp-side h4{margin:8px 0 4px;font-size:12px;color:#aaa;font-weight:600}
.rp-side label{display:flex;align-items:flex-start;gap:6px;margin:3px 0;cursor:pointer;line-height:1.3}
.rp-side label.dis{opacity:.5;cursor:default}
.rp-sw{display:inline-block;width:12px;height:12px;border:1px solid #666;flex:none;margin-top:2px}
.rp-reg{border-top:1px solid #333;padding-top:4px;margin-top:6px}
.rp-reg b{color:#ff9f1a}
.rp-side button{background:#2d2d2d;color:#ddd;border:1px solid #555;border-radius:4px;
  padding:3px 8px;margin:2px 2px 2px 0;cursor:pointer;font-size:12px}
.rp-side button:hover{background:#3a3a3a}
.rp-side input[type=text]{width:100%;box-sizing:border-box;background:#111;color:#eee;
  border:1px solid #555;border-radius:3px;padding:3px 5px;font-size:12px}
.rp-msg{font-size:11px;margin:2px 0 0 18px}
.rp-stage{flex:1;min-width:0;position:relative}
.rp-cv{width:100%;height:100%;display:block;cursor:grab;touch-action:none;background:#000}
.rp-cv.drag{cursor:grabbing}
.rp-cv path{vector-effect:non-scaling-stroke;stroke-width:1px}
.rp-cv polygon{vector-effect:non-scaling-stroke;stroke-width:2px;fill-opacity:.10}
.rp-cv text{font-size:var(--fs,14px);paint-order:stroke;stroke:#000;
  stroke-width:calc(var(--fs,14px)*.3);stroke-linejoin:round;pointer-events:none}
.rp-note{position:absolute;right:8px;bottom:6px;color:#777;font-size:11px;pointer-events:none}
.rp-wrap:fullscreen{height:100vh}
"""

CONFIRM_JS = r"""
export default function(component) {
  const { data, parentElement, setTriggerValue } = component;
  const NS = 'http://www.w3.org/2000/svg';
  let S = parentElement.__rp;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

  if (!S) {
    S = parentElement.__rp = { counter: 0, sig: null, vb: null, vb0: null, hidden: {}, drag: null };
    const wrap = parentElement.querySelector('#rp-wrap');
    wrap.innerHTML =
      '<div class="rp-side" id="rp-side"></div>' +
      '<div class="rp-stage"><svg class="rp-cv" id="rp-cv" xmlns="' + NS + '" preserveAspectRatio="xMidYMid meet">' +
      '<rect id="rp-bg" fill="#000"/><g id="rp-base"></g><g id="rp-ov"></g></svg>' +
      '<div class="rp-note">線幅は拡大しても一定です</div></div>';
    S.wrap = wrap;
    S.svg = parentElement.querySelector('#rp-cv');
    S.side = parentElement.querySelector('#rp-side');
    const svg = S.svg;
    const apply = () => {
      svg.setAttribute('viewBox', S.vb.join(' '));
      const r = svg.getBoundingClientRect();
      if (r.width && r.height)
        svg.style.setProperty('--fs', (Math.max(S.vb[2] / r.width, S.vb[3] / r.height) * 13) + 'px');
    };
    S.apply = apply;
    const pt = (e) => {
      const r = svg.getBoundingClientRect();
      const s = Math.max(S.vb[2] / r.width, S.vb[3] / r.height);
      const offx = (r.width - S.vb[2] / s) / 2, offy = (r.height - S.vb[3] / s) / 2;
      return { x: S.vb[0] + (e.clientX - r.left - offx) * s, y: S.vb[1] + (e.clientY - r.top - offy) * s, s };
    };
    svg.addEventListener('wheel', (e) => {
      e.preventDefault();
      const p = pt(e), k = e.deltaY < 0 ? 0.8 : 1.25;
      const nw = S.vb[2] * k, nh = S.vb[3] * k;
      if (nw > S.vb0[2] * 20 || nw < S.vb0[2] / 2000) return;
      S.vb = [p.x - (p.x - S.vb[0]) * k, p.y - (p.y - S.vb[1]) * k, nw, nh]; apply();
    }, { passive: false });
    svg.addEventListener('pointerdown', (e) => {
      S.drag = { x: e.clientX, y: e.clientY, vb: S.vb.slice(), s: pt(e).s };
      svg.setPointerCapture(e.pointerId); svg.classList.add('drag');
    });
    svg.addEventListener('pointermove', (e) => {
      if (!S.drag) return;
      S.vb = [S.drag.vb[0] - (e.clientX - S.drag.x) * S.drag.s, S.drag.vb[1] - (e.clientY - S.drag.y) * S.drag.s, S.drag.vb[2], S.drag.vb[3]];
      apply();
    });
    const end = () => { S.drag = null; svg.classList.remove('drag'); };
    svg.addEventListener('pointerup', end); svg.addEventListener('pointercancel', end);
    svg.addEventListener('dblclick', () => { S.vb = S.vb0.slice(); apply(); });
    document.addEventListener('fullscreenchange', () => { if (S.render) S.render(); });
  }

  S.data = data;
  S.setTrigger = setTriggerValue;

  const send = (payload) => {
    S.counter += 1;
    S.setTrigger('choice', Object.assign({ seq: Date.now() * 1000 + S.counter }, payload));
  };

  const choose = (rid, kind, idx, text, typedText) => {
    const d = S.data;
    if (d.locked) return;
    const ch = d.choices[String(rid)];
    if (!ch) return;
    // その場で見た目を更新（Python 側の再実行を待たない）
    if (kind === 'cand') {
      if (ch.cands.some((c) => c.idx === idx && c.selected)) return;
      ch.cands.forEach((c) => { c.selected = (c.idx === idx); }); ch.none = false; ch.typed = false;
    } else if (kind === 'none') {
      if (ch.none) return;
      ch.cands.forEach((c) => { c.selected = false; }); ch.none = true; ch.typed = false;
    } else if (kind === 'typed') {
      if (ch.typed) return;
      ch.cands.forEach((c) => { c.selected = false; }); ch.none = false; ch.typed = true;
    } else if (kind === 'typedtext') {
      ch.cands.forEach((c) => { c.selected = false; }); ch.none = false; ch.typed = true; ch.typedtext = typedText;
    }
    send({ region: rid, kind, idx: idx === undefined ? null : idx, text: text || '', typed_text: typedText === undefined ? null : typedText });
    S.render();
  };

  const regColor = (r, C) => r.state === 'decided' || r.state === 'unnamed' ? C.decided : (r.state === 'unnamed_off' ? C.off : C.pending);
  const geomText = (g, col) => {
    const size = g.size;
    const style = size ? 'font-size:' + size.toFixed(2) + 'px;stroke-width:' + (size * 0.25).toFixed(2) + 'px' : '';
    const rot = g.rot ? ' transform="rotate(' + g.rot.toFixed(2) + ' ' + g.x.toFixed(2) + ' ' + g.y.toFixed(2) + ')"' : '';
    return '<text x="' + g.x.toFixed(2) + '" y="' + g.y.toFixed(2) + '" fill="' + col + '" text-anchor="' + g.anchor +
           '" dominant-baseline="' + g.base + '"' + rot + ' style="' + style + '">' + esc(g.text) + '</text>';
  };

  const render = () => {
    const d = S.data, C = d.colors;
    if (d.empty) { S.wrap.innerHTML = '<div style="padding:16px;color:#aaa">描画対象のエンティティがありません</div>'; return; }
    if (S.sig !== d.sig) {
      S.sig = d.sig;
      S.vb0 = [0, 0, d.viewBox[0], d.viewBox[1]]; S.vb = S.vb0.slice();
      S.svg.querySelector('#rp-bg').setAttribute('width', d.viewBox[0]);
      S.svg.querySelector('#rp-bg').setAttribute('height', d.viewBox[1]);
      S.svg.querySelector('#rp-base').innerHTML = d.svg;
    }
    // --- 図面上の重ね表示 ---
    let ov = '';
    d.regions.forEach((r) => {
      if (S.hidden[r.key]) return;
      const col = regColor(r, C);
      ov += '<polygon points="' + r.points.map((p) => p[0].toFixed(2) + ',' + p[1].toFixed(2)).join(' ') +
            '" fill="' + col + '" stroke="' + col + '"/>';
      if (r.state !== 'pending') {
        // 確定した領域の名称は元のラベルの位置に（同じ文字列が複数あればすべて）。無名領域は多角形の左上
        if (r.geoms && r.geoms.length) r.geoms.forEach((g) => { ov += geomText(g, col); });
        else ov += '<text x="' + (r.anchor[0] + 4) + '" y="' + (r.anchor[1] + 16) + '" fill="' + col + '">' + esc(r.name) + '</text>';
      }
    });
    d.regions.forEach((r) => {
      if (r.state !== 'pending' || S.hidden[r.key]) return;
      const ch = d.choices[String(r.id)];
      (ch ? ch.cands : []).forEach((c) => {
        const col = c.selected ? C.candSel : C.candOther;
        (c.geoms || []).forEach((g) => { ov += geomText(g, col); });
      });
    });
    S.svg.querySelector('#rp-ov').innerHTML = ov;
    S.apply();

    // --- 左パネル ---
    const scroll = S.side.scrollTop;
    let h = '<h4>領域</h4>';
    d.regions.forEach((r) => {
      const col = regColor(r, C);
      h += '<label><input type="checkbox" data-vis="' + r.key + '" ' + (S.hidden[r.key] ? '' : 'checked') + '>' +
           '<span class="rp-sw" style="background:' + col + '"></span><span>' + esc(r.label) +
           (r.state !== 'pending' ? '　' + esc(r.name) + (r.state === 'unnamed_off' ? '（特定OFF）' : '') : '') + '</span></label>';
    });
    h += '<h4>抽出領域の選択</h4>';
    const dis = d.locked ? ' dis' : '';
    d.regions.filter((r) => r.state === 'pending').forEach((r) => {
      const ch = d.choices[String(r.id)];
      h += '<div class="rp-reg"><b>' + esc(r.label) + '</b>';
      ch.cands.forEach((c) => {
        h += '<label class="' + dis.trim() + '"><input type="checkbox" data-c="' + r.id + ':' + c.idx + '" ' +
             (c.selected ? 'checked ' : '') + (d.locked ? 'disabled' : '') + '><span class="rp-sw" style="background:' +
             (c.selected ? C.candSel : C.candOther) + '"></span><span>' + esc(c.text) + '（距離 ' + c.dist + '）</span></label>';
      });
      h += '<label class="' + dis.trim() + '"><input type="checkbox" data-n="' + r.id + '" ' + (ch.none ? 'checked ' : '') +
           (d.locked ? 'disabled' : '') + '><span>（領域選択しない）</span></label>';
      h += '<label class="' + dis.trim() + '"><input type="checkbox" data-t="' + r.id + '" ' + (ch.typed ? 'checked ' : '') +
           (d.locked ? 'disabled' : '') + '><span>領域名を入力</span></label>';
      h += '<input type="text" data-tx="' + r.id + '" value="' + esc(ch.typedtext || '') + '" ' + (d.locked ? 'disabled ' : '') +
           'placeholder="図面枠内のラベルと完全一致（全角・小文字可）">';
      if (ch.typed) {
        h += ch.typedMatch ? '<div class="rp-msg" style="color:#7fdc7f">採用: ' + esc(ch.typedMatch) + '</div>'
                           : '<div class="rp-msg" style="color:#e8a">一致するラベルなし（未選択になります）</div>';
      }
      h += '</div>';
    });
    h += '<h4>操作</h4><button data-act="fit">全体表示</button><button data-act="fs">' +
         (document.fullscreenElement ? '全画面を終了' : '全画面') + '</button><div id="rp-fsnote" style="color:#e8a"></div>' +
         '<div style="color:#888;margin-top:6px;line-height:1.5">ホイール: 拡大縮小<br>ドラッグ: 移動<br>ダブルクリック: 全体表示</div>' +
         (d.hasFont ? '' : '<div style="color:#e8a;margin-top:8px">日本語フォントが見つからないため、日本語の文字は表示されません。</div>');
    S.side.innerHTML = h;
    S.side.scrollTop = scroll;

    S.side.querySelectorAll('input[data-vis]').forEach((el) => el.addEventListener('change', () => {
      S.hidden[el.getAttribute('data-vis')] = !el.checked; render();
    }));
    S.side.querySelectorAll('input[data-c]').forEach((el) => el.addEventListener('change', () => {
      const [rid, idx] = el.getAttribute('data-c').split(':').map(Number);
      const c = S.data.choices[String(rid)].cands.find((x) => x.idx === idx);
      if (!el.checked) { render(); return; }
      choose(rid, 'cand', idx, c.text);
    }));
    S.side.querySelectorAll('input[data-n]').forEach((el) => el.addEventListener('change', () => {
      if (!el.checked) { render(); return; }
      choose(Number(el.getAttribute('data-n')), 'none');
    }));
    S.side.querySelectorAll('input[data-t]').forEach((el) => el.addEventListener('change', () => {
      if (!el.checked) { render(); return; }
      choose(Number(el.getAttribute('data-t')), 'typed');
    }));
    S.side.querySelectorAll('input[data-tx]').forEach((el) => el.addEventListener('change', () => {
      choose(Number(el.getAttribute('data-tx')), 'typedtext', undefined, '', el.value);
    }));
    S.side.querySelectorAll('button[data-act]').forEach((b) => b.addEventListener('click', () => {
      const act = b.getAttribute('data-act');
      if (act === 'fit') { S.vb = S.vb0.slice(); S.apply(); return; }
      const note = S.side.querySelector('#rp-fsnote'); note.textContent = '';
      if (document.fullscreenElement) { document.exitFullscreen(); return; }
      const p = S.wrap.requestFullscreen ? S.wrap.requestFullscreen() : null;
      if (!p) { note.textContent = 'この環境では全画面にできません'; return; }
      p.catch(() => { note.textContent = 'この環境では全画面にできません'; });
    }));
  };
  S.render = render;
  render();
}
"""
