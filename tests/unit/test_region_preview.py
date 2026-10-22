"""
このテストが守るもの: 領域のブラウザ内プレビュー（model/region_preview.py）が、
グレーの図面に領域の多角形を正しい位置・色・初期表示で重ね、自己完結HTMLとして返すこと。
合成DXFを中心に検証する（実データ1件のみ slow）。
"""
import inspect
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ezdxf
import pytest
from ezdxf.fonts import font_manager as ezdxf_font_manager
from ezdxf.fonts import fonts as ezdxf_fonts

from model import region_preview as rp
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


def test_confirmation_overlays_all_target_checked_with_label():
    ovs = rp.confirmation_overlays([_reg(2, RECT, frame=1, area_pct=29.6)])
    assert len(ovs) == 1
    assert (ovs[0].style, ovs[0].checked) == ("target", True)
    assert ovs[0].label == "図面2 / 領域3（30%）"


def test_preview_overlays_mark_to_style_and_checked():
    analysis = {"regions": [_reg(0, RECT), _reg(1, RECT), _reg(2, RECT)], "minor_regions": [_reg(7, RECT)]}
    rows = [
        {"id": 0, "name": "A", "area_pct": 10, "mark": "特定"},
        {"id": 1, "name": "B", "area_pct": 10, "mark": ""},
        {"id": 2, "name": "C", "area_pct": 10, "mark": "除外"},
        {"id": 7, "name": "小", "area_pct": 2, "mark": "特定"},  # 小領域も引ける
        {"id": 99, "name": "迷子", "area_pct": 1, "mark": "特定"},  # 多角形なし → 飛ばす
    ]
    got = {o.key: (o.style, o.checked) for o in rp.preview_overlays(analysis, rows)}
    assert got == {"r0": ("target", True), "r1": ("other", False),
                   "r2": ("excluded", False), "r7": ("target", True)}


def test_same_name_in_multiple_frames_keeps_distinct_keys():
    analysis = {"regions": [_reg(0, RECT, frame=0), _reg(1, RECT, frame=1)], "minor_regions": []}
    rows = [{"id": 0, "name": "SAME", "area_pct": 5, "mark": "特定"},
            {"id": 1, "name": "SAME", "area_pct": 5, "mark": "特定"}]
    assert len({o.key for o in rp.preview_overlays(analysis, rows)}) == 2


def test_html_is_self_contained_and_has_overlay_controls():
    base = rp.render_base_svg(_dxf_bytes(with_text=True))
    ovs = [rp.RegionOverlay("r0", "領域<A>", RECT, "target", True),
           rp.RegionOverlay("r1", "B", RECT, "other", False)]
    html_text = rp.build_region_viewer_html(base, ovs, 600)
    assert "<script src" not in html_text
    assert not re.findall(r'(?:src|href)="https?://', html_text)
    assert html_text.count('data-ov="r0"') == 2  # チェックボックスと図面上のグループ
    assert "領域&lt;A&gt;" in html_text  # エスケープされる
    assert 'data-preset="target"' in html_text and 'data-preset="none"' in html_text


def test_empty_overlays_still_draw_drawing():
    base = rp.render_base_svg(_dxf_bytes())
    assert "表示できる領域がありません" in rp.build_region_viewer_html(base, [], 600)


def test_empty_dxf_is_empty_without_error():
    buf = io.StringIO()
    ezdxf.new("R2010").write(buf)
    base = rp.render_base_svg(buf.getvalue().encode("utf-8"))
    assert base.empty
    assert "描画対象のエンティティがありません" in rp.build_region_viewer_html(base, [], 600)


def test_shared_hook_defaults_to_none():
    params = inspect.signature(region_selection.render_region_confirmation_section).parameters
    assert params["file_preview"].default is None


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
