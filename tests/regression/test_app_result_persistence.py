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

実際のブラウザ（Playwright）で app.py を起動し、DXFアップロード→
「ラベルを抽出」の操作後もExcelダウンロードボタンが表示され続けることを
確認するブラックボックステスト。playwright未インストール/ブラウザ未
ダウンロードの環境ではスキップする。
"""
import os
import socket
import subprocess
import sys
import time

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from tests.sample_data import pool_exists, require_sample  # noqa: E402

# sample-dxf 配下をファイル名で探す（無ければ、sample-dxf があるとき収集時に失敗する）
SAMPLE_DXF = require_sample('EE6492-039-38A.dxf')

pytest.importorskip('playwright.sync_api')
from playwright.sync_api import sync_playwright  # noqa: E402

pytestmark = pytest.mark.skipif(not pool_exists(), reason='共有 sample-dxf が無い環境')


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture(scope='module')
def streamlit_server():
    """app.py を別プロセスで起動する。"""
    port = _free_port()
    env = dict(os.environ)

    proc = subprocess.Popen(
        [sys.executable, '-m', 'streamlit', 'run', 'app.py',
         '--server.headless', 'true', '--server.port', str(port)],
        cwd=PROJECT_ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 30
        connected = False
        while time.time() < deadline:
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=1):
                    connected = True
                    break
            except OSError:
                time.sleep(0.5)
        if not connected:
            proc.terminate()
            pytest.skip('streamlit サーバーの起動に失敗（タイムアウト）')
        time.sleep(1.5)  # Streamlit初回スクリプト実行の余裕
        yield f'http://localhost:{port}'
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope='module')
def browser():
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(headless=True)
        except Exception as e:  # ブラウザ未ダウンロード等
            pytest.skip(f'Playwrightブラウザが利用できません: {e}')
        yield b
        b.close()


@pytest.mark.slow
def test_excel_result_persists_after_extract(streamlit_server, browser):
    """通常モード（既定）+「端子一覧を抽出」ON、「領域を検出」OFF（既定）で
    「ラベルを抽出」を押した直後もExcelダウンロードボタンが表示され続けること
    （消えずに残ること）を確認する。

    v3.0.0で「図面番号・タイトル・サブタイトルを抽出」オプションは廃止された
    （常に抽出されるようになった）ため、このテストではチェックしない。"""
    page = browser.new_page(viewport={'width': 1400, 'height': 1000})
    try:
        page.goto(streamlit_server, wait_until='networkidle')
        page.wait_for_timeout(1000)

        page.locator('input[type="file"]').set_input_files(SAMPLE_DXF)
        page.wait_for_timeout(1500)

        # Streamlit のチェックボックスはネイティブ input を画面外へ視覚的に隠し、
        # 独自スタイルの box（同じ label 内の兄弟要素）を表示する実装のため、
        # input への click()（force=True でも）は要素がビューポート外にあり失敗する。
        # input に click イベントを直接ディスパッチすることで、実際のマウス座標に
        # 依存せず確実にチェック状態を切り替える。
        page.get_by_role('checkbox', name='端子一覧を抽出').dispatch_event('click')
        page.wait_for_timeout(300)

        page.get_by_role('button', name='ラベルを抽出').click()
        page.wait_for_timeout(6000)

        assert page.get_by_text('エラーが発生しました', exact=False).count() == 0, (
            'ラベル抽出後にエラーが表示された')
        assert page.get_by_role('button', name='抽出結果をダウンロード').count() > 0, (
            'ラベル抽出後にExcelダウンロードボタンが表示されない'
            '（excel_resultが「領域を検出」OFFガードで誤って消されるバグの再発）')
    finally:
        page.close()


def test_region_mode_excel_result_produced(streamlit_server, browser):
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
    page = browser.new_page(viewport={'width': 1400, 'height': 1000})
    try:
        page.goto(streamlit_server, wait_until='networkidle')
        page.wait_for_timeout(1000)

        page.locator('input[type="file"]').set_input_files(SAMPLE_DXF)
        page.wait_for_timeout(1500)

        page.get_by_role('checkbox', name='領域を検出').dispatch_event('click')
        page.wait_for_timeout(500)

        # exact=True: 素の name マッチは部分一致のため、チェックボックス脇の
        # ヘルプアイコン（アクセシブルネーム "Help for 領域を検出"）とも
        # 曖昧一致してしまう。
        page.get_by_role('button', name='領域を検出', exact=True).click()
        page.wait_for_timeout(4000)

        # v3.18.0: 「抽出領域の選択」「領域選択を完了」は廃止。検出直後に「領域の確認」が出て、
        # そのまま「ラベルを抽出」に進める
        assert page.get_by_text('領域の確認', exact=False).count() > 0, (
            '領域の確認セクションが表示されていない（テストの前提が崩れている）')

        page.get_by_role('button', name='ラベルを抽出').click()
        page.wait_for_timeout(6000)

        assert page.get_by_text('エラーが発生しました', exact=False).count() == 0, (
            'ラベル抽出後にエラーが表示された（領域検出モードの回帰）')
        assert page.get_by_role('button', name='抽出結果をダウンロード').count() > 0, (
            '領域検出モードでExcelダウンロードボタンが表示されない')
    finally:
        page.close()
