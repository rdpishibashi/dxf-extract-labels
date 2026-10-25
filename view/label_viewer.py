"""抽出結果プレビューの「図面を表示」viewer（streamlit 依存のビュー層。DXF-extract-labels 専用）。

「機器符号候補」の選択（DEF／CAN／（空欄））に当たるラベルを、図面上で元の文字の色を変えて示す表示専用の部品
（選択を Streamlit へ返さない）。図面の描画は「領域の確認」viewer と共通（`region_confirm_viewer.render_region_base()`
の `st.cache_data`。同じ DXF なら1回だけ描く）。

構成:
  - viewer 用データ: `model/label_preview.py`（streamlit 非依存）
  - 部品の HTML/CSS/JS: `view/assets/label_viewer.{html,css,js}`（初めて使うときに読む。インストーラーに同梱すること）
"""
from functools import lru_cache
from pathlib import Path

import streamlit as st

from model import label_preview
from view import region_confirm_viewer

LABEL_VIEWER_HEIGHT_PX = region_confirm_viewer.REGION_CONFIRM_HEIGHT_PX
ASSET_DIR = Path(__file__).resolve().parent / 'assets'
_ASSET_FILES = {
    'LABEL_HTML': 'label_viewer.html',
    'LABEL_CSS': 'label_viewer.css',
    'LABEL_JS': 'label_viewer.js',
}


@lru_cache(maxsize=None)
def _read_asset(name):
    return (ASSET_DIR / _ASSET_FILES[name]).read_text(encoding='utf-8')


def __getattr__(name):
    """`LABEL_HTML`/`LABEL_CSS`/`LABEL_JS`（viewer 部品の定義）を、使われたときに読む。"""
    if name in _ASSET_FILES:
        return _read_asset(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# 部品の定義（HTML/CSS/JS）が変わったときだけ登録し直す（同名の再登録は警告になる。
# `region_confirm_viewer._get_region_confirm_viewer` と同じ理由）。
@st.cache_resource(show_spinner=False)
def _get_label_viewer(definition_sig):
    return st.components.v2.component(
        "label_viewer",
        html=_read_asset('LABEL_HTML'),
        css=_read_asset('LABEL_CSS'),
        js=_read_asset('LABEL_JS'),
    )


def mount_viewer(**kwargs):
    """viewer 部品を描く。テストは、この関数を差し替えて渡された data を調べる。"""
    sig = hash((_read_asset('LABEL_HTML'), _read_asset('LABEL_CSS'), _read_asset('LABEL_JS')))
    return _get_label_viewer(sig)(**kwargs)


def render_label_viewer(fname, dxf_bytes, positions, mark_of, selected_marks, key):
    """図面 `dxf_bytes` を描き、`selected_marks`（区分の集合）に当たるラベルを色付けする。"""
    try:
        base = region_confirm_viewer.render_region_base(dxf_bytes)
        data = label_preview.build_label_viewer_data(base, fname, positions, mark_of, selected_marks)
        data['height'] = LABEL_VIEWER_HEIGHT_PX  # viewer の外枠の高さ（JS が明示的に設定する）
        mount_viewer(key=key, data=data, height=LABEL_VIEWER_HEIGHT_PX)
    except Exception as e:
        st.warning(f"図面を表示できませんでした: {e}")
