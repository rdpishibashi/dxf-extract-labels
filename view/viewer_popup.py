"""viewer（`st.components.v2` の部品）の「別ウィンドウ」ボタン用の共有部品（streamlit 非依存）。

別ウィンドウは閲覧専用（拡大縮小・移動・全体表示だけ）。部品の JS が、ボタンを押した時点の図面の SVG を
複製して新しいウィンドウに書き込む。ズームは全体表示に戻る。窓の CSS／JS は `view/assets/viewer_popup.{css,js}`
で、部品の HTML 定義の末尾に `<template id="<prefix>-popup">` として同梱する（部品の JS が読み出す）。
"""
from functools import lru_cache
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent / 'assets'


@lru_cache(maxsize=None)
def _read(name):
    return (ASSET_DIR / name).read_text(encoding='utf-8')


def popup_template(prefix):
    """部品の HTML 定義に足す `<template>`（`prefix` は部品の ID 接頭辞。`rp`／`lv`）。"""
    return (f'<template id="{prefix}-popup"><style>{_read("viewer_popup.css")}</style>'
            f'<script>{_read("viewer_popup.js")}</script></template>')


def definition_html(base_html, prefix):
    """部品の HTML 定義（`base_html`）に、別ウィンドウ用の `<template>` を足したもの。"""
    return base_html + popup_template(prefix)
