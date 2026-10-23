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
from dataclasses import dataclass
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


@dataclass
class RegionOverlay:
    key: str
    label: str
    corners: List[Tuple[float, float]]  # DXF（モデル空間）座標
    style: str = "target"  # 'target' | 'other' | 'excluded'
    checked: bool = True


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
    )


def _to_svg(matrix: Tuple[float, ...], x: float, y: float) -> Tuple[float, float]:
    v = Matrix44(matrix).transform(Vec3(x, y, 0))
    return float(v.x), float(v.y)


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
        overlays.append(RegionOverlay(
            key=f"r{row['id']}",
            label=f"{name}（{row.get('area_pct', 0):.0f}%）",
            corners=corners,
            style=style,
            checked=checked,
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
#cv .ov text{font-size:14px;paint-order:stroke;stroke:#000;stroke-width:3px;
  stroke-linejoin:round;pointer-events:none}
#note{position:absolute;right:8px;bottom:6px;color:#777;font-size:11px;pointer-events:none}
"""

# pan/zoom は model/dxf_preview.py（DXF-diff-manager 正本）の JS を複製
_VIEWER_JS = """
(function(){
  var svg=document.getElementById('cv');
  var vb0=svg.getAttribute('viewBox').split(' ').map(Number);
  var vb=vb0.slice();
  function apply(){svg.setAttribute('viewBox',vb.join(' '));}
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
    # ラベルは多角形の左上付近（SVG 座標で最小 y の頂点のうち最小 x）
    tx, ty = min(pts, key=lambda p: (p[1], p[0]))
    dash = ' stroke-dasharray="6 4"' if ov.style == "excluded" else ""
    return (
        f'<g class="ov" data-ov="{html.escape(ov.key)}">'
        f'<polygon points="{points}" fill="{color}" stroke="{color}"{dash}/>'
        f'<text x="{tx + 4:.2f}" y="{ty + 16:.2f}" fill="{color}">{html.escape(ov.label)}</text></g>'
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
