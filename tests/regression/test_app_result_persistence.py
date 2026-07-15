"""app.py の抽出結果セッション状態が消えない回帰テスト。

検証済み不具合（2026-07-15ユーザー報告）:
  「図面番号・タイトル・サブタイトルを抽出」+「端子一覧を抽出」をONにし、
  「領域を検出」はOFF（既定）のまま「ラベルを抽出」→「未確定ラベル」で
  「選択完了」を押すと、Excelファイルが出力されず「ラベルを抽出」ボタンの
  初期表示に戻ってしまう。

根本原因: 「領域を検出」チェックボックスOFF時、前回検出済みの
`region_analyses`（stale data）を消すガード処理（`app.py` の
`if not enable_region_detection:` ブロック）が、OFF中の**毎回の再描画**で
実行される。v1.9.4でこのブロックに `_REGION_DETECT_CLEAR_KEYS`
（`excel_result`/`is_region_mode`/`region_results_summary`）を誤って
含めてしまったため、「選択完了」や「ラベルを抽出」で `excel_result` を
セットした直後の再描画（`st.rerun()`後）で即座に消えていた
（`region_analyses` 以外は消してはならない）。

実際のブラウザ（Playwright）で app.py を起動し、DXFアップロード→
「ラベルを抽出」→「選択完了」の一連の操作の後もExcelダウンロードボタンが
表示され続けることを確認するブラックボックステスト。playwright未インストール
/ ブラウザ未ダウンロードの環境ではスキップする。
"""
import os
import socket
import subprocess
import sys
import time

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_DXF = os.path.join(
    PROJECT_ROOT, 'sample-dxf', 'terminal-detector', 'EE6492-039-38A.dxf')

pytest.importorskip('playwright.sync_api')
from playwright.sync_api import sync_playwright  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(SAMPLE_DXF),
    reason='sample-dxf/terminal-detector/EE6492-039-38A.dxf が無い')


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture(scope='module')
def streamlit_server(tmp_path_factory):
    """app.py を別プロセスで起動する。判断ログは一時ディレクトリへリダイレクトし、
    ユーザーの実 `~/Documents/DXF-extract-labels/decision_log.csv` を汚さない。"""
    port = _free_port()
    log_path = tmp_path_factory.mktemp('decision_log') / 'decision_log.csv'
    env = dict(os.environ)
    env['DXF_DECISION_LOG_PATH'] = str(log_path)

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


def test_excel_result_persists_after_select_complete(streamlit_server, browser):
    """候補モード（既定）+「図面番号・タイトル・サブタイトルを抽出」+
    「端子一覧を抽出」ON、「領域を検出」OFF（既定）で抽出→選択完了しても、
    Excelダウンロードボタンが表示され続けること（消えずに残ること）を確認する。"""
    page = browser.new_page(viewport={'width': 1400, 'height': 1000})
    try:
        page.goto(streamlit_server, wait_until='networkidle')
        page.wait_for_timeout(1000)

        page.locator('input[type="file"]').set_input_files(SAMPLE_DXF)
        page.wait_for_timeout(1500)

        page.get_by_text('図面番号・タイトル・サブタイトルを抽出', exact=False).click()
        page.get_by_text('端子一覧を抽出', exact=False).click()
        page.wait_for_timeout(300)

        page.get_by_role('button', name='ラベルを抽出').click()
        page.wait_for_timeout(6000)

        assert page.get_by_text('未確定ラベル', exact=False).count() > 0, (
            '未確定ラベルセクションが表示されていない（テストの前提が崩れている）')

        page.get_by_role('button', name='選択完了').click()
        page.wait_for_timeout(6000)

        assert page.get_by_text('エラーが発生しました', exact=False).count() == 0, (
            '選択完了後にエラーが表示された')
        assert page.get_by_role('button', name='Excelをダウンロード').count() > 0, (
            '選択完了後にExcelダウンロードボタンが表示されない'
            '（excel_resultが「領域を検出」OFFガードで誤って消されるバグの再発）')
    finally:
        page.close()
