"""app.py の抽出結果セッション状態が消えない回帰テスト。

検証済み不具合（2026-07-15ユーザー報告）:
  「図面番号・タイトル・サブタイトルを抽出」+「端子一覧を抽出」をONにし、
  「領域を検出」はOFF（既定）のまま「ラベルを抽出」を押すと、Excelファイルが
  出力されず「ラベルを抽出」ボタンの初期表示に戻ってしまう。

根本原因: 「領域を検出」チェックボックスOFF時、前回検出済みの
`region_analyses`（stale data）を消すガード処理（`app.py` の
`if not enable_region_detection:` ブロック）が、OFF中の**毎回の再描画**で
実行される。v1.9.4でこのブロックに `_REGION_DETECT_CLEAR_KEYS`
（`excel_result`/`is_region_mode`/`region_results_summary`）を誤って
含めてしまったため、「ラベルを抽出」で `excel_result` をセットした直後の
再描画（`st.rerun()`後）で即座に消えていた（`region_analyses` 以外は
消してはならない）。

**2026-07-30 判定条件の簡素化**: 「未確定ラベル」UIでのレビュー・
「選択完了」ボタンは廃止され、「ラベルを抽出」を押した時点で直ちに
Excelが生成されるようになった（`model/ref_designator.py` 参照）。本テストも
これに合わせ、「ラベルを抽出」直後にExcelダウンロードボタンが表示され
続けることを確認する形に更新した。

Streamlit の `AppTest`（ブラウザもサーバーも使わずに app.py を実行する）で、
DXFアップロード→「ラベルを抽出」の操作後に、さらに再描画してもExcel
ダウンロードボタンが表示され続けることを確認するブラックボックステスト。

**2026-11-06 Playwright から AppTest へ置き換え**（ユーザー決定: Playwright は
入っていない前提にし、skip をなくす）。以前は実ブラウザで app.py を操作して
おり、Playwright が無い環境では skip されていた（venv の作り直しで Playwright
が抜け、skip が 51→52 件になって発覚）。`AppTest` のファイルアップロード操作
（`file_uploader[...].set_value()`）は streamlit 1.67 で確認した。
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.sample_data import pool_exists, require_sample  # noqa: E402

# sample-dxf 配下をファイル名で探す（無ければ、sample-dxf があるとき収集時に失敗する）
SAMPLE_DXF = require_sample('EE6492-039-38A.dxf')

pytestmark = pytest.mark.skipif(not pool_exists(), reason='共有 sample-dxf が無い環境')

APP_PATH = os.path.join(PROJECT_ROOT, 'app.py')


def _app_with_sample_uploaded():
    """app.py を起動し、サンプル DXF をアップロードした状態の AppTest を返す。"""
    at = AppTest.from_file(APP_PATH, default_timeout=120).run()
    with open(SAMPLE_DXF, 'rb') as f:
        content = f.read()
    at.file_uploader[0].set_value(
        (os.path.basename(SAMPLE_DXF), content, 'application/octet-stream')).run()
    return at


def _checkbox(at, label):
    return next(c for c in at.checkbox if c.label == label)


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _assert_download_shown(at, message):
    assert not at.exception, f'例外が発生した: {[e.value for e in at.exception]}'
    assert not [e for e in at.error if 'エラーが発生しました' in str(e.value)], (
        'ラベル抽出後にエラーが表示された')
    assert any(d.label == '抽出結果をダウンロード' for d in at.download_button), message


@pytest.mark.slow
def test_excel_result_persists_after_extract():
    """通常モード（既定）+「端子一覧を抽出」ON、「領域を検出」OFF（既定）で
    「ラベルを抽出」を押した直後もExcelダウンロードボタンが表示され続けること
    （消えずに残ること）を確認する。

    v3.0.0で「図面番号・タイトル・サブタイトルを抽出」オプションは廃止された
    （常に抽出されるようになった）ため、このテストではチェックしない。"""
    at = _app_with_sample_uploaded()
    _checkbox(at, '端子一覧を抽出').check().run()
    _button(at, 'ラベルを抽出').click().run()
    _assert_download_shown(at, 'ラベル抽出後にExcelダウンロードボタンが表示されない')

    # 抽出直後の再描画で excel_result が消えていた不具合（「領域を検出」OFF のガード）を
    # 捕まえるため、何も操作せずにもう一度再描画しても残ることを確かめる
    at.run()
    _assert_download_shown(
        at, '再描画後にExcelダウンロードボタンが消えた'
            '（excel_resultが「領域を検出」OFFガードで誤って消されるバグの再発）')


def test_region_mode_excel_result_produced():
    """領域検出モード（「領域を検出」ON）で、領域検出→領域選択を完了→
    ラベル抽出までエラーなくExcel出力されることを確認する。

    v3.0.0で「機器符号（候補）以外も抽出」オプションは廃止され、領域検出モードは
    常に機器符号抽出パイプライン（`ref_designator.build_ref_designator_region_results()`）
    を経由するようになった。

    `app.py` の 809行の単一 `app()` 関数を `_render_intro`/`_render_upload_section`/
    `_render_options_section`/`_render_region_config_form`/
    `_render_region_detection_trigger`/`_render_global_region_names_section`/
    `_render_region_confirmation_section`/`_handle_extract_button`/
    `_render_results_section` へ分割した際（2026-07-26、モジュラリティ改善）の
    回帰確認。`test_excel_result_persists_after_extract` は「領域を検出」OFFの
    既定パスのみを通るため、本テストは領域検出モード側の関数群
    （`_render_region_config_form`・`_render_region_detection_trigger`・
    `_render_global_region_names_section`・`_render_region_confirmation_section`）
    を実際に経由する。

    2026-10-25（v3.18.0）更新: 「領域一覧」（抽出領域の選択）と「領域選択を完了」は廃止した。
    検出後は何も操作せず「ラベルを抽出」まで進めることを確認する。
    """
    at = _app_with_sample_uploaded()
    _checkbox(at, '領域を検出').check().run()
    _button(at, '領域を検出').click().run()

    # v3.18.0: 「抽出領域の選択」「領域選択を完了」は廃止。検出直後に「領域の確認」が出て、
    # そのまま「ラベルを抽出」に進める
    texts = [m.value for m in at.markdown] + [h.value for h in at.subheader] + [h.value for h in at.header]
    assert any('領域の確認' in (t or '') for t in texts), (
        '領域の確認セクションが表示されていない（テストの前提が崩れている）')

    _button(at, 'ラベルを抽出').click().run()
    _assert_download_shown(at, '領域検出モードでExcelダウンロードボタンが表示されない')
