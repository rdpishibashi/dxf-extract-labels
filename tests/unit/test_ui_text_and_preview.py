"""ユーザー指示4件（2026-09-11）の UI 検証。

1. 「領域の確認」セクション: 図面枠が見つからない既知の未対応ケース
   （`_FRAME_NOT_FOUND_ERROR_MARKER`）は表示しない。
2. 「DXFファイルのアップロード」セクションのラベル文言変更
   （「複数可」→「複数可，フォルダ可」）。
3. 「抽出結果統計」（集計テーブル）→「抽出結果プレビュー」（Excelシート選択+表示、
   DXF-diff-manager と同じパターン）への置き換え。
4. ダウンロードボタンの文言変更（「Excelをダウンロード」→「抽出結果をダウンロード」）。

`app.py` は Streamlit のウィジェットを直接描画する関数群のため、
`streamlit.testing.v1.AppTest.from_string()` で実際にスクリプトとして実行し、
描画結果を検証する（`test_region_list_no_candidates_message.py` と同じ理由で
`from_function()` ではなく `from_string()` を使う——pytest全体実行時のみ
再現するクロージャ経由の session_state 汚染を避けるため）。
"""
import base64
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402
from model.excel_output import create_ref_designator_excel_output  # noqa: E402
from app import _FRAME_NOT_FOUND_ERROR_MARKER  # noqa: E402


def _minimal_excel_bytes():
    """`create_ref_designator_excel_output()` が要求する最小限のフィールドを
    持つ合成データから、実物と同じ構造のExcelバイト列を作る
    （実DXFファイルへの依存を避け、テストを高速・自己完結にするため）。"""
    ref_final = {
        'a.dxf': {
            'rows': [{'機器符号候補': 'Y', 'ラベル': 'R10', '個数': 1}],
            'total_in_frame': 1, 'ref_designator_count': 1, 'frames': 1,
            'warning': None, 'main_drawing_number': 'EE1234-500-01A',
            'source_drawing_number': None, 'title': 'TEST', 'subtitle': None,
        }
    }
    return create_ref_designator_excel_output(ref_final, sort_option='asc')


# ---------------------------------------------------------------------------
# 1. 「領域の確認」— 図面枠が見つからない既知の未対応ケースを表示しない
# ---------------------------------------------------------------------------

_CONFIRM_SCRIPT_TEMPLATE = '''
import sys
sys.path.insert(0, {project_root!r})
import streamlit as st
from app import _render_region_confirmation_section
st.session_state['region_analyses'] = {analyses}
st.session_state['region_selection_confirmed'] = True
_render_region_confirmation_section(enable_region_detection=True, locked=False)
'''


def _run_confirmation(analyses):
    # Python の repr() で埋め込む（json.dumps() だと None が 'null' になり
    # 埋め込み先のPythonソースとして不正になるため）。
    script = _CONFIRM_SCRIPT_TEMPLATE.format(
        project_root=PROJECT_ROOT, analyses=repr(analyses))
    at = AppTest.from_string(script)
    at.run()
    return at


def test_frame_not_found_file_not_shown_in_confirmation():
    """図面枠が見つからない（`_FRAME_NOT_FOUND_ERROR_MARKER`を含むエラー）
    ファイルは、「領域の確認」セクションに一切表示されない
    （ファイル名見出し・警告メッセージのいずれも出さない）。"""
    err = '図面枠（太さ 100 の線で囲まれた枠）が見つからなかったため、' + \
        _FRAME_NOT_FOUND_ERROR_MARKER
    at = _run_confirmation({
        'no_frame.dxf': {'error': err, 'regions': [], 'frames': []},
    })
    assert at.exception.len == 0
    all_text = [m.value for m in at.markdown] + [w.value for w in at.warning]
    assert not any('no_frame.dxf' in str(t) for t in all_text)
    assert not any(err in str(t) for t in all_text)


def test_other_error_still_shown_in_confirmation():
    """図面枠検出以外の genuine な例外（`str(e)`経由のエラー）は、
    従来どおりファイル名・警告メッセージともに表示される
    （抑制は図面枠未検出ケースのみに限定されることの担保）。"""
    err = 'DXFファイルの読み込みに失敗しました: corrupt file'
    at = _run_confirmation({
        'broken.dxf': {'error': err, 'regions': [], 'frames': []},
    })
    assert at.exception.len == 0
    markdown_texts = [m.value for m in at.markdown]
    warning_texts = [w.value for w in at.warning]
    assert any('broken.dxf' in str(t) for t in markdown_texts)
    assert any(err == str(t) for t in warning_texts)


def test_confirmation_unaffected_when_no_error():
    """エラーが無く、要選択の領域も無いファイルは従来どおり非表示のまま
    （既存仕様への回帰が無いことの担保）。"""
    at = _run_confirmation({
        'clean.dxf': {'error': None, 'regions': [], 'frames': [(0, 100, 0, 100)]},
    })
    assert at.exception.len == 0
    assert at.markdown.len == 0


# ---------------------------------------------------------------------------
# 2. 「DXFファイルのアップロード」— ラベル文言変更
# ---------------------------------------------------------------------------

_UPLOAD_SCRIPT = '''
import sys
sys.path.insert(0, {project_root!r})
from app import _render_upload_section
_render_upload_section()
'''.format(project_root=PROJECT_ROOT)


def test_upload_label_mentions_folder_support():
    at = AppTest.from_string(_UPLOAD_SCRIPT)
    at.run()
    assert at.exception.len == 0
    labels = [w.label for w in at.get('file_uploader')]
    assert any('複数可，フォルダ可' in lbl for lbl in labels)
    assert not any(lbl.endswith('（複数可）') for lbl in labels)


# ---------------------------------------------------------------------------
# 3・4. 抽出結果セクション — プレビュー化・ダウンロードボタン文言
# ---------------------------------------------------------------------------

_RESULTS_SCRIPT_TEMPLATE = '''
import sys
sys.path.insert(0, {project_root!r})
import base64
import streamlit as st
from app import _render_results_section
st.session_state['excel_result'] = base64.b64decode({excel_b64!r})
st.session_state['output_filename'] = 'extracted_labels.xlsx'
st.session_state['is_region_mode'] = False
st.session_state['ref_results_summary'] = {{'a.dxf': {{}}}}
_render_results_section()
'''


def _run_results_section():
    excel_b64 = base64.b64encode(_minimal_excel_bytes()).decode('ascii')
    script = _RESULTS_SCRIPT_TEMPLATE.format(
        project_root=PROJECT_ROOT, excel_b64=excel_b64)
    at = AppTest.from_string(script)
    at.run()
    return at


def test_results_section_shows_preview_not_statistics():
    """「抽出結果統計」（集計テーブル）は表示されず、「抽出結果プレビュー」
    （Excelシート選択+表示）に置き換わっていることを固定する。"""
    at = _run_results_section()
    assert at.exception.len == 0
    expander_labels = [e.label for e in at.expander]
    assert '抽出結果プレビュー' in expander_labels
    assert not any('抽出結果統計' in lbl for lbl in expander_labels)


def test_results_section_preview_lists_actual_excel_sheets():
    """プレビューのシート選択肢が、実際に生成したExcelファイルのシート名
    （Summary・抽出ラベル一覧・ファイル別シート）と一致することを固定する
    （集計値の再計算ではなく、本物のExcelバイト列を読んでいることの担保）。
    ファイル別シート名は拡張子除去済み（'a.dxf' → 'a'、既存のシート名仕様）。"""
    at = _run_results_section()
    assert at.exception.len == 0
    selectboxes = [sb for sb in at.selectbox if sb.key == 'extract_preview_sheet']
    assert len(selectboxes) == 1
    assert selectboxes[0].options == ['Summary', '抽出ラベル一覧', 'a']
    # 選択中シートの内容がdataframeとして表示されていること
    assert at.dataframe.len >= 1


def test_download_button_label_changed():
    """ダウンロードボタンの文言が「抽出結果をダウンロード」であり、
    旧文言「Excelをダウンロード」ではないことを固定する。"""
    at = _run_results_section()
    assert at.exception.len == 0
    button_labels = [b.label for b in at.download_button]
    assert '抽出結果をダウンロード' in button_labels
    assert 'Excelをダウンロード' not in button_labels
