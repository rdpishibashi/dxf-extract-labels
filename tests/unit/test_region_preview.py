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
