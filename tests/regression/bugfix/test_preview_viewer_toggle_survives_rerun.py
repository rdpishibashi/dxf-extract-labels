"""通常モード（「領域を検出」OFF）で、プレビューの「図面を表示」トグルが再実行後も ON のまま残る。

不具合ID: 2026-11-09 EE7319-300-01A.dxf の報告（「図面を表示」を ON にしても viewer が出ない）。
以前の壊れ方: 「領域を検出」OFF の間は、`app.py` が再実行のたびに `_REGION_DETECT_CLEAR_PREFIXES`
でキーを消していた。この接頭辞に `extract_preview_viewer_`（「図面を表示」トグル）が入っていたため、
ON にした直後の再実行で False に戻り、viewer が一度も描かれなかった（v3.17.0 以来、どの図面でも）。
修正後に保証すること: OFF 中の毎回の消去は `_REGION_OFF_CLEAR_PREFIXES`（トグルを含まない）を使い、
トグルが ON のまま保たれ、`mount_viewer` が呼ばれる。シート選択で pandas の `ExcelFile.parse`
非推奨警告（Pandas4Warning）も出ない。

`tests/unit/test_label_preview.py` の AppTest は `from_string` の小さなスクリプトで、`app.py` 本体の
「OFF 中の毎回の消去」を通らないため、この不具合を検出できなかった。ここは `app.py` 本体を動かす。
"""
import os
import sys
import warnings

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.sample_data import pool_exists, require_sample  # noqa: E402
from view import label_viewer  # noqa: E402

SAMPLE_DXF = require_sample('EE6492-039-38A.dxf')
SHEET = 'EE6492-039-38A'
APP_PATH = os.path.join(PROJECT_ROOT, 'app.py')

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not pool_exists(), reason='共有 sample-dxf が無い環境')]


@pytest.fixture
def mounted(monkeypatch):
    """`label_viewer.mount_viewer` を記録用に差し替える（app.py と同じプロセスなので効く）。"""
    calls = []
    monkeypatch.setattr(label_viewer, 'mount_viewer', lambda **kw: calls.append(kw['data']))
    return calls


def _extracted_app():
    """サンプルをアップロードして「ラベルを抽出」を押し、図面のシートを選んだ AppTest。"""
    at = AppTest.from_file(APP_PATH, default_timeout=300).run()
    with open(SAMPLE_DXF, 'rb') as f:
        content = f.read()
    at.file_uploader[0].set_value(
        (os.path.basename(SAMPLE_DXF), content, 'application/octet-stream')).run()
    next(b for b in at.button if b.label == 'ラベルを抽出').click().run()
    next(s for s in at.selectbox if s.label == 'シートを選択').set_value(SHEET).run()
    return at


def _toggle(at):
    return next(t for t in at.toggle if t.label == '図面を表示')


def test_viewer_toggle_stays_on_and_draws_in_normal_mode(mounted):
    """通常モードで「図面を表示」を ON にすると、再実行後も ON のまま viewer が描かれる。"""
    at = _extracted_app()
    assert not mounted  # OFF のうちは描かない
    _toggle(at).set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    assert _toggle(at).value is True, '再実行でトグルが False に戻った（OFF 中の消去がトグルを含んでいる）'
    assert len(mounted) == 1 and mounted[0]['svg']
    at.run()  # 何も操作しない再実行でも保たれる
    assert _toggle(at).value is True
    assert len(mounted) == 2


def test_sheet_selection_raises_no_pandas_parse_deprecation():
    """シート選択で `ExcelFile.parse` の非推奨警告が出ない（`pd.read_excel` を使う）。"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _extracted_app()
    messages = [str(w.message) for w in caught if 'ExcelFile.parse' in str(w.message)]
    assert not messages, messages
