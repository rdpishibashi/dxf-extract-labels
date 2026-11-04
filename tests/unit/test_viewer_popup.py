"""
このテストが守るもの: viewer（領域の確認・図面を表示）の「別ウィンドウ」ボタンの部品（view/viewer_popup.py）。
- 部品の HTML 定義に、窓用の CSS／JS を入れた `<template>` が同梱される（部品の JS がここから読む）
- 部品の JS が、ボタン（data-act="pop"）・開く処理・`<template>` の参照先（#<prefix>-popup）を持つ
- 窓の JS が、部品の JS が複製した SVG に付ける印（data-popup-cv）を探す
動作そのもの（実際にウィンドウが開く・拡大縮小・全体表示）はブラウザでの確認が必要で、ここでは扱わない。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest  # noqa: E402

from view import label_viewer as lv  # noqa: E402
from view import region_confirm_viewer as rcv  # noqa: E402
from view import viewer_popup  # noqa: E402

_VIEWERS = [("lv", lv.LABEL_HTML, lv.LABEL_JS), ("rp", rcv.CONFIRM_HTML, rcv.CONFIRM_JS)]


@pytest.mark.parametrize("prefix,base_html,js", _VIEWERS)
def test_definition_html_carries_popup_template(prefix, base_html, js):
    html = viewer_popup.definition_html(base_html, prefix)
    assert html.startswith(base_html)
    assert f'<template id="{prefix}-popup">' in html
    assert "<style>" in html and "<script>" in html


@pytest.mark.parametrize("prefix,base_html,js", _VIEWERS)
def test_viewer_js_opens_popup_from_its_template(prefix, base_html, js):
    assert 'data-act="pop"' in js
    assert f"#{prefix}-popup" in js
    assert "data-popup-cv" in js
    assert "window.open(" in js


def test_popup_script_finds_the_cloned_svg_and_cannot_close_the_template_early():
    js = viewer_popup._read("viewer_popup.js")
    assert "data-popup-cv" in js
    # <template> の中に <script> として入れるので、文字列中に </script> を含めてはいけない
    assert "</script" not in js.lower()
    assert "</style" not in viewer_popup._read("viewer_popup.css").lower()
