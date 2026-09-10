"""「領域一覧」セクションで、面積条件を満たす領域が1件も無かったときの
案内メッセージを検証する（2026-09-10、ユーザー指示による文言変更）。

旧UI（「領域検出の詳細設定」フォームがあった頃）の「面積条件を満たす領域が
検出されませんでした。詳細設定の調整をお試しください。」は、詳細設定UIの
撤去（config.py化）に伴い「一定面積の領域は検出されませんでした。」へ変更した。

`app._render_global_region_names_section()` は Streamlit のウィジェットを
直接描画する関数のため、`streamlit.testing.v1.AppTest.from_string()` で
実際にスクリプトとして実行し、描画された `st.info` の文言を検証する
（`tests/regression/test_app_result_persistence.py` が使う Playwright 経由の
ブラウザテストより軽量——ブラウザ起動不要でファイルアップロードも介さない）。

**`from_function()` ではなく `from_string()` を使う理由**: 当初
`AppTest.from_function()`（`inspect.getsourcelines()` でクロージャのソースを
取得して実行）を使っていたが、`tests/unit/` を一括実行すると本ファイル内の
3番目のテストだけが「実行していないはずの前段テストの session_state」を
引き継いだ状態で失敗する現象が発生した（本ファイル単体では再現しない・
最小再現コードでも再現しない、pytest 全体実行時のみ発生する不安定な現象）。
原因を特定するよりも、クロージャ経由の間接実行を避けて `AppTest.from_string()`
（文字列ソースを直接スクリプトとして実行、closure・inspect 一切不使用）に
切り替える方が確実なため変更した。
"""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402

_SCRIPT_TEMPLATE = '''
import sys
sys.path.insert(0, {project_root!r})
import streamlit as st
from app import _render_global_region_names_section
{session_state_setup}
_render_global_region_names_section(locked=False)
'''


def _run(region_analyses_regions_by_file=None):
    """`_render_global_region_names_section()` を、指定した
    `{ファイル名: [region dict, ...]}` を `region_analyses` として
    session_state にセットした状態で AppTest 上で実行する。
    `None` の場合は session_state に何もセットしない
    （「領域を検出」未実行の状態を模す）。"""
    if region_analyses_regions_by_file is None:
        setup = ''
    else:
        analyses = {
            fname: {'regions': regions}
            for fname, regions in region_analyses_regions_by_file.items()
        }
        setup = f"st.session_state['region_analyses'] = {json.dumps(analyses)}"
    script = _SCRIPT_TEMPLATE.format(project_root=PROJECT_ROOT, session_state_setup=setup)
    at = AppTest.from_string(script)
    at.run()
    return at


def test_no_regions_shows_new_message():
    """全ファイルで検出領域が0件（＝面積条件を満たす領域が無かった）とき、
    新文言の st.info が表示される。"""
    at = _run({'a.dxf': []})
    assert at.exception.len == 0
    info_texts = [i.value for i in at.info]
    assert info_texts == ['一定面積の領域は検出されませんでした。']
    # 旧UI（詳細設定への言及）の文言が残っていないことも確認する
    assert not any('詳細設定' in t for t in info_texts)


def test_regions_without_name_candidates_also_shows_message():
    """検出領域自体は存在するが、いずれも名称候補（name_candidates）が
    1件も無い場合も、同じメッセージが表示される（`collect_unique_region_
    candidate_names()` の戻り値が空になる条件はこのケースも含むため）。"""
    at = _run({'a.dxf': [{'id': 0, 'name_candidates': []}]})
    assert at.exception.len == 0
    info_texts = [i.value for i in at.info]
    assert info_texts == ['一定面積の領域は検出されませんでした。']


def test_no_message_when_region_analyses_absent():
    """「領域を検出」自体をまだ実行していない（session_stateに region_analyses
    が無い）場合はメッセージを出さない（早期returnのまま。誤って毎回表示され
    ないことの担保）。"""
    at = _run(None)
    assert at.exception.len == 0
    assert [i.value for i in at.info] == []


def test_no_message_when_candidates_exist():
    """名称候補が実際に存在する場合はメッセージを出さず、「領域一覧」の
    見出しが表示される（既存の正常系が壊れていないことの担保）。"""
    at = _run({'a.dxf': [{'id': 0, 'name_candidates': [[1.0, 'B CHAMBER']]}]})
    assert at.exception.len == 0
    assert [i.value for i in at.info] == []
    assert any(sh.value == '領域一覧' for sh in at.subheader)
