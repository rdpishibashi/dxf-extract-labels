"""
このテストが守るもの: 領域のブラウザ内プレビュー（model/region_preview.py）が、
グレーの図面に領域の多角形を正しい位置・色・初期表示で重ね、自己完結HTMLとして返すこと。
合成DXFを中心に検証する（実データ1件のみ slow）。
"""
import inspect
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ezdxf
import pytest
from ezdxf.fonts import font_manager as ezdxf_font_manager
from ezdxf.fonts import fonts as ezdxf_fonts

from model import region_preview as rp
from view import region_confirm_viewer as rcv
from view import region_selection


def _dxf_bytes(with_text=False) -> bytes:
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (100, 0), (100, 50), (0, 50)], close=True)
    if with_text:
        msp.add_text("検図", dxfattribs={"insert": (5, 5), "height": 3})
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")


def _reg(rid, corners, frame=0, area_pct=10.0):
    return {"id": rid, "frame": frame, "corners": corners, "polygon": corners, "area_pct": area_pct}


RECT = [(10, 10), (40, 10), (40, 30), (10, 30)]


def test_overlay_svg_coordinates_inside_viewbox_with_y_flip():
    base = rp.render_base_svg(_dxf_bytes())
    w, h = base.view_box
    pts = [rp._to_svg(base.matrix, x, y) for x, y in RECT]
    assert all(0 <= x <= w and 0 <= y <= h for x, y in pts)
    (x0, y0), (x1, y1) = pts[0], pts[2]  # 左下 → 右上
    assert x1 > x0
    assert y1 < y0  # SVG は y 下向き


def test_empty_dxf_is_empty_without_error():
    buf = io.StringIO()
    ezdxf.new("R2010").write(buf)
    base = rp.render_base_svg(buf.getvalue().encode("utf-8"))
    assert base.empty


@pytest.fixture
def no_system_fonts(monkeypatch):
    manager = ezdxf_fonts.font_manager
    saved, saved_fb = manager.dumps(), manager._fallback_font_name
    saved_defaults = list(ezdxf_font_manager.DEFAULT_FONTS)
    manager.clear()
    manager._fallback_font_name = ""
    monkeypatch.setitem(ezdxf_font_manager.FONT_DIRECTORIES, manager.platform, [])
    yield manager
    manager.clear()
    manager.loads(saved)
    manager._fallback_font_name = saved_fb
    ezdxf_font_manager.DEFAULT_FONTS[:] = saved_defaults


def test_renders_japanese_text_without_system_fonts(no_system_fonts):
    """Streamlit Cloud 相当（システムフォント無し）でも同梱フォントで描ける。"""
    assert (rp.BUNDLED_FONT_DIR / rp.BUNDLED_FONT_NAME).is_file()
    base = rp.render_base_svg(_dxf_bytes(with_text=True))
    assert base.cjk_font == rp.BUNDLED_FONT_NAME
    assert not base.empty


def test_table_hook_is_opt_in_and_viewer_has_fullscreen_button():
    """「抽出領域の選択」の表の上のフック（DXF-extract-labels が「すべての領域名を特定にする」ボタンに使う）は
    既定で何も出さず、viewer に全画面ボタンがある。"""
    params = inspect.signature(region_selection.render_global_region_names_section).parameters
    assert params["above_table"].default is None
    assert 'data-act="fs"' in rcv.CONFIRM_JS


# ---------------------------------------------------------------------------
# 元の文字の色替え（v3.17.0）: 文字ごとの CSS クラス・白一色の見た目・パス座標
# ---------------------------------------------------------------------------

def _label_doc():
    """直置き・ブロック内（同じブロックを2回）・入れ子ブロック・色指定付き MTEXT・非表示・寸法を含む合成 DXF。"""
    doc = ezdxf.new("R2010")
    doc.layers.add("OFF", color=1).off()
    b1 = doc.blocks.new("B1")
    b1.add_line((0, 0), (5, 0))
    b1.add_text("ALPHA", dxfattribs={"insert": (1, 1), "height": 2})
    outer = doc.blocks.new("OUTER")
    outer.add_blockref("B1", (0, 10))
    outer.add_text("OMEGA", dxfattribs={"insert": (2, 2), "height": 2})
    b2 = doc.blocks.new("B2")
    b2.add_text("HIDDEN", dxfattribs={"insert": (1, 1), "height": 2})  # 画層 0（親 INSERT の画層で描かれる）
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (120, 0), (120, 60), (0, 60)], close=True)
    msp.add_text("DIRECT", dxfattribs={"insert": (50, 5), "height": 2})
    msp.add_blockref("B1", (20, 20))
    msp.add_blockref("B1", (60, 20))
    msp.add_blockref("OUTER", (30, 30))
    msp.add_blockref("B2", (80, 5), dxfattribs={"layer": "OFF"})
    msp.add_mtext("\\C1;COLORED", dxfattribs={"insert": (70, 40), "char_height": 2})
    msp.add_mtext("AB{\\C1;CD}{\\c255;EF}", dxfattribs={"insert": (70, 50), "char_height": 2})  # 途中に色指定が複数
    msp.add_text("GHOST", dxfattribs={"insert": (90, 40), "height": 2, "invisible": 1})
    msp.add_linear_dim(base=(0, -8), p1=(0, -2), p2=(30, -2)).render()  # 寸法値は INSERT から参照されないブロックの文字
    return doc


def _to_bytes(doc) -> bytes:
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")


def _path_ds(svg_text):
    import re
    from collections import Counter
    return Counter(re.findall(r'\bd="([^"]*)"', svg_text))


def _reference_path_ds(dxf_bytes):
    """文字を移す前（従来）と同じ描き方: 全体を白一色の ColorPolicy.CUSTOM で描いたときのパス。"""
    from ezdxf import recover
    from ezdxf.addons.drawing import Frontend, RenderContext, config, layout, svg
    doc, _ = recover.read(io.BytesIO(dxf_bytes))
    cjk = rp.find_cjk_font()
    if cjk:
        for style in doc.styles:
            style.dxf.font = cjk
            if style.dxf.hasattr("bigfont"):
                style.dxf.bigfont = ""
    cfg = config.Configuration(background_policy=config.BackgroundPolicy.BLACK,
                               color_policy=config.ColorPolicy.CUSTOM, custom_fg_color="#ffffff")
    backend = svg.SVGBackend()
    Frontend(RenderContext(doc), backend, config=cfg).draw_layout(doc.modelspace())
    return _path_ds(backend.get_string(layout.Page(0, 0, layout.Units.mm)))


def _classes_by_text(base):
    out = {}
    for (text, _x, _y), classes in base.label_classes.items():
        out.setdefault(text, []).append(classes)
    return out


def test_every_visible_label_gets_its_own_css_class():
    """守るもの: 直置き・ブロック内（同じブロックの複数回の INSERT・入れ子）・色指定付き MTEXT の文字が、
    1件ずつ別の CSS クラスを持つ（個別に色を変えられる）。非表示の文字（invisible 属性・画層 OFF の INSERT 内の
    画層 0 の文字）はクラスを持たない（ezdxf も描かない）。"""
    base = rp.render_base_svg(_to_bytes(_label_doc()))
    by_text = _classes_by_text(base)
    assert len(by_text["ALPHA"]) == 3  # 直置きの INSERT 2回＋入れ子（OUTER 内）1回
    assert all(len(v) == 1 for v in by_text.values() if v is not by_text["ALPHA"])
    assert {"DIRECT", "OMEGA", "COLORED", "ABCDEF"} <= set(by_text)  # 途中の色指定があっても1ラベル・1クラス
    assert "HIDDEN" not in by_text and "GHOST" not in by_text
    all_cls = [cls for v in base.label_classes.values() for cls, _kind in v]
    assert len(all_cls) == len(set(all_cls))  # クラスはラベルごとに別
    assert all(kind == "fill" for v in base.label_classes.values() for _cls, kind in v)


def test_marker_colors_do_not_leak_and_paths_match_the_plain_render():
    """守るもの: 目印の色は SVG に残らず（見た目は従来どおり白一色）、文字を移しても全パスの座標が従来の描画と一致する
    （寸法線の寸法値を含む。全ブロックから文字を消すと寸法値が消えてここで落ちる）。"""
    import re
    data = _to_bytes(_label_doc())
    base = rp.render_base_svg(data)
    assert not re.search(r"#8[0-9a-fA-F]{5}\b", base.inner_svg)
    assert _path_ds(base.inner_svg) == _reference_path_ds(data)


def test_minsert_lifts_every_grid_element():
    """守るもの: MINSERT（格子状の INSERT）の文字が、格子の要素ごとに別クラスで描かれる
    （`virtual_entities()` は先頭の要素しか返さないため `multi_insert()` で分ける）。"""
    doc = ezdxf.new("R2010")
    blk = doc.blocks.new("GRID")
    blk.add_text("CELL", dxfattribs={"insert": (1, 1), "height": 2})
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (100, 0), (100, 50), (0, 50)], close=True)
    msp.add_blockref("GRID", (10, 10), dxfattribs={
        "row_count": 2, "column_count": 3, "row_spacing": 10, "column_spacing": 20})
    data = _to_bytes(doc)
    base = rp.render_base_svg(data)
    assert len(_classes_by_text(base)["CELL"]) == 6
    assert _path_ds(base.inner_svg) == _reference_path_ds(data)


def test_label_classes_of_returns_classes_or_empty():
    """守るもの: `label_classes_of()` は NFKC 正規化したキーで引き、図面に無いラベルは空（重ね書きにフォールバック）。"""
    base = rp.render_base_svg(_to_bytes(_label_doc()))
    (text, x, y), classes = next((k, v) for k, v in base.label_classes.items() if k[0] == "DIRECT")
    assert rp.label_classes_of(base, (text, x, y)) == classes
    assert rp.label_classes_of(base, ("ＤＩＲＥＣＴ", x, y)) == classes  # 全角でも同じキー
    assert rp.label_classes_of(base, ("NOPE", 1, 1)) == []


SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def test_region_label_occurrences_prefers_inside_labels_only():
    """守るもの: 領域名の強調は、その領域の内側の同名ラベルすべて。領域から離れた同名（図面情報枠など）は出さない。"""
    labels = [("RACK", 10, 10), ("rack", 50, 50), ("RACK", 300, 300), ("OTHER", 5, 5)]
    got = rp.region_label_occurrences("RACK", labels, SQUARE)
    assert [l[1:] for l in got] == [(10, 10), (50, 50)]


def test_region_label_occurrences_falls_back_to_nearest_when_none_inside():
    """守るもの: 内側に無ければ最も近い1件（入力名・親の候補を引き継ぐ小領域のため）。一致が無ければ空。"""
    labels = [("RACK", 300, 300), ("RACK", 150, 10)]
    assert [l[1:] for l in rp.region_label_occurrences("RACK", labels, SQUARE)] == [(150, 10)]
    assert rp.region_label_occurrences("NONE", labels, SQUARE) == []
    assert len(rp.region_label_occurrences("RACK", labels, [])) == 2  # 多角形が無ければ絞らない
