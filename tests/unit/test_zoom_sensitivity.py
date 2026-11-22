"""
このテストが守るもの: DXF-extract-labels の viewer（領域の確認・図面を表示・別ウィンドウ）のズーム感度の計算部が、
ホイール・タッチパッド・ピンチの回転量に比例して拡大縮小すること、感度の保存が壊れた環境でも既定値で動くこと、
左パネルと別ウィンドウにスライダーがあること。

計算部は `// zoom-sens:begin` 〜 `// zoom-sens:end` で囲まれ、3 つの JS（label_viewer.js・region_confirm_viewer.js・viewer_popup.js）、
DXF-diff-manager の `model/dxf_preview.py`、Cable-checker の viewer に同じ内容がある（変えるときは全部そろえる）。
囲まれた部分だけを Node の vm で実行して検証する（ブラウザ不要）。ブラウザでの動き（保存・別ウィンドウ）は実機で確認する。
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT.parent
ASSETS = ROOT / "view" / "assets"
MODULES = [ASSETS / "label_viewer.js", ASSETS / "region_confirm_viewer.js"]
POPUP = ASSETS / "viewer_popup.js"
JS_FILES = MODULES + [POPUP]

NODE = shutil.which("node")
BEGIN, END = "// zoom-sens:begin", "// zoom-sens:end"

_RUNNER = r"""
const vm = require('vm');
const fs = require('fs');
const block = fs.readFileSync(process.argv[2], 'utf8');
const spec = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
function load(ls) {
  const c = { localStorage: ls, Math, Number, isFinite, String };
  vm.createContext(c); vm.runInContext(block, c); return c;
}
const out = { factors: [], stored: {}, throws: {} };
const mem = new Map();
const okLs = { getItem: (k) => (mem.has(k) ? mem.get(k) : null), setItem: (k, v) => mem.set(k, String(v)) };
const c0 = load(okLs);
for (const f of spec.factors) out.factors.push(c0.zoomFactor(f.e, f.sens));
out.default = c0.getZoomSens();
for (const v of spec.stored) { mem.clear(); mem.set(c0.ZOOM_SENS_KEY, v); out.stored[v] = c0.getZoomSens(); }
mem.clear(); c0.setZoomSens(0.5); out.written = mem.get(c0.ZOOM_SENS_KEY);
const bad = { getItem() { throw new Error('denied'); }, setItem() { throw new Error('denied'); } };
for (const [name, ls] of [['throwing', bad], ['undefined', undefined]]) {
  const c = load(ls);
  const before = c.getZoomSens(); c.setZoomSens(0.7);
  out.throws[name] = [before, c.getZoomSens()];
}
console.log(JSON.stringify(out));
"""

# (イベント, 感度, 期待する倍率)。倍率は viewBox の幅に掛ける値（>1 で縮小表示）
FACTOR_CASES = [
    ({"deltaY": 100, "deltaMode": 0, "ctrlKey": False}, 1.0, 1.2214),     # マウス1ノッチ（現行の 1.25 倍とほぼ同じ）
    ({"deltaY": -100, "deltaMode": 0, "ctrlKey": False}, 1.0, 0.8187),
    ({"deltaY": 5, "deltaMode": 0, "ctrlKey": False}, 1.0, 1.0101),       # タッチパッドの細かいイベント
    ({"deltaY": -3, "deltaMode": 0, "ctrlKey": True}, 1.0, 0.9704),       # ピンチ（ctrlKey 付き）
    ({"deltaY": 3, "deltaMode": 1, "ctrlKey": False}, 1.0, 1.2712),       # 行単位（Firefox の1ノッチ）
    ({"deltaY": 1, "deltaMode": 2, "ctrlKey": False}, 1.0, 1.6487),       # ページ単位は上限で止まる
    ({"deltaY": 1000, "deltaMode": 0, "ctrlKey": False}, 1.0, 1.6487),    # 大きな値は上限 exp(0.5)
    ({"deltaY": -1000, "deltaMode": 0, "ctrlKey": False}, 1.0, 0.6065),
    ({"deltaY": 0, "deltaMode": 0, "ctrlKey": False}, 1.0, 1.0),          # 横スクロールだけのとき変化なし
    ({"deltaY": 100, "deltaMode": 0, "ctrlKey": False}, 0.5, 1.1052),     # 感度に比例
    ({"deltaY": 100, "deltaMode": 0, "ctrlKey": False}, 3.0, 1.6487),     # 感度3.0は上限で止まる
    ({"deltaY": 100, "deltaMode": 0, "ctrlKey": False}, 0.1, 1.0202),
]
STORED_CASES = {"0.5": 0.5, "9": 3.0, "-1": 0.1, "abc": 1.0, "1.04": 1.0}


def _block(js_text):
    assert js_text.count(BEGIN) == 1 and js_text.count(END) == 1
    return js_text[js_text.index(BEGIN): js_text.index(END) + len(END)]


@pytest.fixture(scope="module", params=JS_FILES, ids=[p.name for p in JS_FILES])
def node_result(request, tmp_path_factory):
    if NODE is None:
        pytest.skip("node が見つからない（ズームの計算部は Node でしか検証できない）")
    d = tmp_path_factory.mktemp("zoom")
    (d / "block.js").write_text(_block(request.param.read_text(encoding="utf-8")), encoding="utf-8")
    (d / "runner.js").write_text(_RUNNER, encoding="utf-8")
    spec = {"factors": [{"e": e, "sens": s} for e, s, _ in FACTOR_CASES], "stored": list(STORED_CASES)}
    (d / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    proc = subprocess.run([NODE, str(d / "runner.js"), str(d / "block.js"), str(d / "spec.json")],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_zoom_factor_is_proportional_to_wheel_amount(node_result):
    """イベントごとの倍率が回転量・感度に比例し（上限あり）、deltaMode・ctrlKey を換算する。"""
    for (event, sens, expected), got in zip(FACTOR_CASES, node_result["factors"]):
        assert got == pytest.approx(expected, abs=1e-4), (event, sens)


def test_stored_sensitivity_is_clamped_and_default_is_one(node_result):
    """保存値は 0.1〜3.0 に丸める。値が無いときは 1.0、不正値も 1.0。書き込みは文字列で行う。"""
    assert node_result["default"] == 1.0
    for stored, expected in STORED_CASES.items():
        assert node_result["stored"][stored] == expected, stored
    assert node_result["written"] == "0.5"


def test_sensitivity_works_without_usable_local_storage(node_result):
    """localStorage が例外を投げる・存在しない環境でも、メモリの値で動く（設定したあとは設定値を返す）。"""
    for name in ("throwing", "undefined"):
        before, after = node_result["throws"][name]
        assert before == 1.0 and after == 0.7, name




def test_zoom_block_is_identical_across_viewer_files():
    """3 つの JS の計算部が文字単位で同じ（片方だけ直すと感度の挙動がずれるため）。"""
    blocks = {p.name: _block(p.read_text(encoding="utf-8")) for p in JS_FILES}
    assert len(set(blocks.values())) == 1, {k: len(v) for k, v in blocks.items()}


def test_zoom_block_is_identical_to_diff_manager_viewer():
    """DXF-diff-manager の viewer（正本 dxf_preview.py）の計算部と同じ。隣のプロジェクトが無いときだけ skip。"""
    other = TOOLS / "DXF-diff-manager" / "model" / "dxf_preview.py"
    if not other.parent.is_dir():
        pytest.skip("DXF-diff-manager が隣にない")
    assert _block(other.read_text(encoding="utf-8")) == _block(MODULES[0].read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", MODULES, ids=[p.name for p in MODULES])
def test_module_viewer_has_slider_in_panel_and_popup_and_no_fixed_factor(path):
    """部品の JS は、左パネルと別ウィンドウ用の左パネルの両方にスライダーを出して配線し、旧い固定倍率を使わない。"""
    js = path.read_text(encoding="utf-8")
    assert js.count("zoomSliderHtml() +") == 2          # render() の左パネルと openPopup() の窓
    assert js.count("ホイール・ピンチ: 拡大縮小") == 2
    assert "bindZoomSlider(S.side);" in js
    assert "zoomFactor(e, getZoomSens())" in js
    assert "0.8 : 1.25" not in js


def test_popup_script_binds_slider_reads_saved_value_and_has_no_fixed_factor():
    """別ウィンドウの JS は、窓の中のスライダー（data-zoom-sens）に保存値を入れて配線し、旧い固定倍率を使わない。"""
    js = POPUP.read_text(encoding="utf-8")
    assert "[data-zoom-sens]" in js and "zsShow(getZoomSens())" in js and "setZoomSens(zs.value)" in js
    assert "zoomFactor(e,getZoomSens())" in js and "0.8:1.25" not in js
    assert "</script" not in js.lower()                   # <template> の中の <script> を途中で閉じない
