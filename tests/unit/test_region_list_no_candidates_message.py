"""**v3.18.0（2026-10-25）で DXF-extract-labels の画面から「抽出領域の選択」セクションを削除した**ため、
`app._render_global_region_names_section()` を描く AppTest の5本はスキップ（代替: `test_always_specify_regions.py`）。
共有関数 `region_list_dataframe()` を直接呼ぶテストは、DXF-label-comparison・HostPL-comparer が使い続けるため残す。

「領域一覧」（2026-10-17 に見出しを「抽出領域の選択」へ改称）セクションで、面積条件を満たす領域が1件も無かったときの
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

import pytest

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


@pytest.mark.skip(reason="2026-10-25 v3.18.0: 「抽出領域の選択」セクション削除（アプリは表を描かない）。"
                         "代替: tests/unit/test_always_specify_regions.py")
def test_no_regions_shows_new_message():
    """全ファイルで検出領域が0件（＝面積条件を満たす領域が無かった）とき、
    新文言の st.info が表示される。"""
    at = _run({'a.dxf': []})
    assert at.exception.len == 0
    info_texts = [i.value for i in at.info]
    assert info_texts == ['一定面積の領域は検出されませんでした。']
    # 旧UI（詳細設定への言及）の文言が残っていないことも確認する
    assert not any('詳細設定' in t for t in info_texts)


@pytest.mark.skip(reason="2026-10-25 v3.18.0: 「抽出領域の選択」セクション削除（アプリは表を描かない）。"
                         "代替: tests/unit/test_always_specify_regions.py")
def test_regions_without_name_candidates_are_listed_not_reported_as_missing():
    """検出領域は存在するが、いずれも名称候補が1件も無い場合（2026-10-10 変更）:
    名称のない領域も「領域一覧」に行として出すため、「検出されませんでした」の
    メッセージは出さず、「領域一覧」の見出しを表示する。
    （以前は同じメッセージを出していたが、領域は検出されているので不正確だった。）"""
    at = _run({'a.dxf': [{'id': 0, 'name_candidates': []}]})
    assert at.exception.len == 0
    assert [i.value for i in at.info] == []
    assert any(sh.value == '抽出領域の選択' for sh in at.subheader)


@pytest.mark.skip(reason="2026-10-25 v3.18.0: 「抽出領域の選択」セクション削除（アプリは表を描かない）。"
                         "代替: tests/unit/test_always_specify_regions.py")
def test_no_message_when_region_analyses_absent():
    """「領域を検出」自体をまだ実行していない（session_stateに region_analyses
    が無い）場合はメッセージを出さない（早期returnのまま。誤って毎回表示され
    ないことの担保）。"""
    at = _run(None)
    assert at.exception.len == 0
    assert [i.value for i in at.info] == []


@pytest.mark.skip(reason="2026-10-25 v3.18.0: 「抽出領域の選択」セクション削除（アプリは表を描かない）。"
                         "代替: tests/unit/test_always_specify_regions.py")
def test_no_message_when_candidates_exist():
    """名称候補が実際に存在する場合はメッセージを出さず、「領域一覧」の
    見出しが表示される（既存の正常系が壊れていないことの担保）。"""
    at = _run({'a.dxf': [{'id': 0, 'name_candidates': [[1.0, 'B CHAMBER']]}]})
    assert at.exception.len == 0
    assert [i.value for i in at.info] == []
    assert any(sh.value == '抽出領域の選択' for sh in at.subheader)


def test_locked_region_list_shows_current_selection_not_the_frozen_base():
    """結果表示中（ロック）の「抽出領域の選択」表は、固定してある初期値（base）ではなく現在の選択状態を
    表示する。ロックで `disabled` が変わるとウィジェットIDが変わり編集内容が失われるため、base のままだと
    チェックが全部外れて見え、さらに差分として選択を消してしまっていた（2026-10-17）。"""
    from view.region_selection import region_list_dataframe
    base = {'A': (False, False), 'B': (False, False)}
    current = {'A': (True, False), 'B': (False, True)}
    unlocked = region_list_dataframe(['A', 'B'], base, True)
    locked = region_list_dataframe(['A', 'B'], current, True)
    assert unlocked['特定'].tolist() == [False, False]
    assert locked['特定'].tolist() == [True, False] and locked['除外'].tolist() == [False, True]
    assert list(region_list_dataframe(['A'], base, False).columns) == ['特定', '領域名']


@pytest.mark.skip(reason="2026-10-25 v3.18.0: 「抽出領域の選択」セクション削除（アプリは表を描かない）。"
                         "代替: tests/unit/test_always_specify_regions.py")
def test_apply_to_all_files_option_is_not_shown():
    """守るもの: 「抽出領域の選択」に、オプション「すべてのファイルに適用する」のチェックボックスが出ない
    （2026-10-25 v3.17.2 で削除。5%未満の小領域の採用は、手動選択した名称も含めて常にこの動作）。
    名称候補がある領域があっても（以前はこのとき表の右に出ていた）、チェックボックスも
    セッション状態のキー `minor_include_confirmed` も作られない。"""
    at = _run({'a.dxf': [{'id': 0, 'name_candidates': [[1.0, 'SYSTEM A']]}]})
    assert at.exception.len == 0
    assert [c.label for c in at.checkbox if 'すべてのファイルに適用' in (c.label or '')] == []
    assert 'minor_include_confirmed' not in at.session_state
