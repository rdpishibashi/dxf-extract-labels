"""ユーザー指示4件（2026-09-11）＋「領域の確認」空表示抑制（2026-09-10）の UI 検証。

1. 「領域の確認」セクション: 図面枠が見つからない既知の未対応ケース
   （`_FRAME_NOT_FOUND_ERROR_MARKER`）は表示しない。
2. 「DXFファイルのアップロード」セクションのラベル文言変更
   （「複数可」→「複数可，フォルダ可」）。
3. 「抽出結果統計」（集計テーブル）→「抽出結果プレビュー」（Excelシート選択+表示、
   DXF-diff-manager と同じパターン）への置き換え。
4. ダウンロードボタンの文言変更（「Excelをダウンロード」→「抽出結果をダウンロード」）。
5. 「領域の確認」は表示対象ファイルが1件も無い場合、見出し・キャプション・
   「すべての領域が…決定済みです。」を含め一切描画しない（2026-09-10）。
   判定は純粋関数 `_region_confirmation_targets()` に切り出し、
   `_render_results_section()` には「抽出結果」という見出しを新設した。

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
from app import _FRAME_NOT_FOUND_ERROR_MARKER, _region_confirmation_targets  # noqa: E402


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
_render_region_confirmation_section(enable_region_detection=True, locked={locked})
'''


def _run_confirmation(analyses, locked=False):
    # Python の repr() で埋め込む（json.dumps() だと None が 'null' になり
    # 埋め込み先のPythonソースとして不正になるため）。
    script = _CONFIRM_SCRIPT_TEMPLATE.format(
        project_root=PROJECT_ROOT, analyses=repr(analyses), locked=locked)
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


def test_confirmation_section_fully_hidden_when_no_targets():
    """表示対象ファイルが1件も無い場合（＝全領域が「領域一覧」で決定済み）、
    「領域の確認」の見出し・キャプション・「すべての領域が…決定済みです。」の
    いずれも描画しない（2026-09-10、セクション自体の空表示を抑制）。"""
    at = _run_confirmation({
        'clean.dxf': {'error': None, 'regions': [], 'frames': [(0, 100, 0, 100)]},
    })
    assert at.exception.len == 0
    assert at.subheader.len == 0
    assert at.caption.len == 0
    assert at.info.len == 0
    assert not any('決定済み' in str(i.value) for i in at.info)


def test_confirmation_section_fully_hidden_when_no_targets_and_locked():
    """結果表示中（`locked=True`、振り返り専用状態）でも、表示対象ファイルが
    1件も無ければ同様に何も描画しない（組み合わせ表の状態A確認）。"""
    at = _run_confirmation({
        'clean.dxf': {'error': None, 'regions': [], 'frames': [(0, 100, 0, 100)]},
    }, locked=True)
    assert at.exception.len == 0
    assert at.subheader.len == 0
    assert at.caption.len == 0
    assert at.info.len == 0


# ---------------------------------------------------------------------------
# 5. `_region_confirmation_targets()` — 表示対象ファイルの判定（純粋関数）
# ---------------------------------------------------------------------------


def test_targets_empty_when_no_pending_and_no_error():
    """要選択領域が0件・エラーも無いファイルは対象に含まれない。"""
    analyses = {
        'clean.dxf': {'error': None, 'regions': []},
    }
    assert _region_confirmation_targets(analyses, checked_names=set()) == []


def test_targets_empty_when_only_frame_not_found_error():
    """図面枠未検出の既知エラーのみのファイルは対象に含まれない。"""
    err = '図面枠が見つからなかったため、' + _FRAME_NOT_FOUND_ERROR_MARKER
    analyses = {
        'no_frame.dxf': {'error': err, 'regions': []},
    }
    assert _region_confirmation_targets(analyses, checked_names=set()) == []


def test_targets_include_genuine_error_even_with_no_pending_regions():
    """要選択領域が0件でも、genuine なエラーを持つファイルは対象に含まれる
    （エラー表示の唯一の場所であるため）。"""
    err = 'DXFファイルの読み込みに失敗しました: corrupt file'
    analyses = {
        'broken.dxf': {'error': err, 'regions': []},
    }
    targets = _region_confirmation_targets(analyses, checked_names=set())
    assert len(targets) == 1
    fname, analysis, pending_regions = targets[0]
    assert fname == 'broken.dxf'
    assert pending_regions == []


def test_targets_include_files_with_pending_regions():
    """チェック済み候補が2件以上残る領域を持つファイルは対象に含まれる。"""
    reg = {
        'id': 0, 'frame': 0, 'area_pct': 50.0, 'corners': [],
        'name_candidates': [(1.0, 'A'), (2.0, 'B')],
    }
    analyses = {
        'contested.dxf': {'error': None, 'regions': [reg]},
    }
    targets = _region_confirmation_targets(analyses, checked_names={'A', 'B'})
    assert len(targets) == 1
    fname, analysis, pending_regions = targets[0]
    assert fname == 'contested.dxf'
    assert pending_regions == [reg]


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


def test_results_section_has_extraction_results_subheader():
    """「領域の確認」を空のときに表示しなくした（2026-09-10）ことに伴い、
    抽出結果プレビュー等の置き場として「抽出結果」という見出しの
    セクションが表示されることを固定する。"""
    at = _run_results_section()
    assert at.exception.len == 0
    subheader_values = [s.value for s in at.subheader]
    assert '抽出結果' in subheader_values


# ---------------------------------------------------------------------------
# 「結線図-組立図を比較」オプション（v3.6.0新設）
# ---------------------------------------------------------------------------

_OPTIONS_SCRIPT = '''
import sys
sys.path.insert(0, {project_root!r})
from app import _render_options_section
_render_options_section(locked=False)
'''.format(project_root=PROJECT_ROOT)


def test_options_section_has_compare_assembly_checkbox_after_terminal_option():
    """「結線図-組立図を比較」チェックボックスが「端子一覧を抽出」の直後に
    配置されていることを固定する（ユーザー指定の配置）。"""
    at = AppTest.from_string(_OPTIONS_SCRIPT)
    at.run()
    assert at.exception.len == 0
    labels = [c.label for c in at.checkbox]
    assert '結線図-組立図を比較' in labels
    assert labels.index('結線図-組立図を比較') == labels.index('端子一覧を抽出') + 1


_RESULTS_SCRIPT_WITH_EXTRA_STATE_TEMPLATE = '''
import sys
sys.path.insert(0, {project_root!r})
import base64
import streamlit as st
from app import _render_results_section
st.session_state['excel_result'] = base64.b64decode({excel_b64!r})
st.session_state['output_filename'] = 'extracted_labels.xlsx'
st.session_state['is_region_mode'] = False
st.session_state['ref_results_summary'] = {{'a.dxf': {{}}}}
{extra_state}
_render_results_section()
'''


def _run_results_section_with_extra_state(extra_state: str):
    excel_b64 = base64.b64encode(_minimal_excel_bytes()).decode('ascii')
    script = _RESULTS_SCRIPT_WITH_EXTRA_STATE_TEMPLATE.format(
        project_root=PROJECT_ROOT, excel_b64=excel_b64, extra_state=extra_state)
    at = AppTest.from_string(script)
    at.run()
    return at


def test_results_section_shows_assembly_compare_warning_when_no_wiring_files():
    """A（結線図）側が1件も見つからない場合、`assembly_compare_warning` の
    文言が警告として表示される（D4）。"""
    at = _run_results_section_with_extra_state(
        "st.session_state['assembly_compare_warning'] = 'A側の警告テスト文言'")
    assert at.exception.len == 0
    warning_values = [w.value for w in at.warning]
    assert 'A側の警告テスト文言' in warning_values


def test_results_section_shows_b_candidate_selection_ui():
    """Bを自動確定できない場合（0件・複数件）、候補一覧の selectbox と
    「比較を実行」ボタンが表示される（D3）。"""
    extra_state = (
        "st.session_state['assembly_compare_candidates'] = ['b.dxf', 'c.dxf']\n"
        "st.session_state['extract_results_full'] = {'a.dxf': {'rows': []}}\n"
        "st.session_state['extract_sort_value'] = 'asc'"
    )
    at = _run_results_section_with_extra_state(extra_state)
    assert at.exception.len == 0
    selectboxes = [sb for sb in at.selectbox if sb.key == 'assembly_compare_b_select']
    assert len(selectboxes) == 1
    assert selectboxes[0].options == ['b.dxf', 'c.dxf']
    button_labels = [b.label for b in at.button]
    assert '比較を実行' in button_labels


def test_results_section_no_assembly_compare_ui_when_not_applicable():
    """比較OFF（`assembly_compare_candidates`/`assembly_compare_warning` が
    いずれもセットされていない）場合、比較関連のUIは一切表示されない。"""
    at = _run_results_section()
    assert at.exception.len == 0
    assert not any(sb.key == 'assembly_compare_b_select' for sb in at.selectbox)
    assert not any(b.label == '比較を実行' for b in at.button)


# ---------------------------------------------------------------------------
# `_resolve_assembly_compare()` — B自動判定の分岐（app.py 引き継ぎ書 §5 表2）
#
# `st.session_state` への書き込みを伴うため、`AppTest.from_string()` で実行
# して結果を検証する（model層の純関数〈select_wiring_files 等〉は
# tests/unit/test_assembly_compare.py で検証済み。ここでは分岐とセッション
# キーへの反映のみを対象にする）。
# ---------------------------------------------------------------------------

_RESOLVE_SCRIPT_TEMPLATE = '''
import sys
sys.path.insert(0, {project_root!r})
import streamlit as st
from app import _resolve_assembly_compare
results = {results!r}
result = _resolve_assembly_compare(results, {option})
st.session_state['_test_result'] = result
'''


def _run_resolve(results, compare_assembly_option=True):
    script = _RESOLVE_SCRIPT_TEMPLATE.format(
        project_root=PROJECT_ROOT, results=results, option=compare_assembly_option)
    at = AppTest.from_string(script)
    at.run()
    return at


def test_resolve_assembly_compare_off_returns_none():
    """比較OFF（`compare_assembly_option=False`）の場合、A/B判定を行わず
    常に None を返す。"""
    results = {'a.dxf': {'rows': [], 'title': 'UNIT内結線図'}}
    at = _run_resolve(results, compare_assembly_option=False)
    assert at.exception.len == 0
    assert at.session_state['_test_result'] is None


def test_resolve_assembly_compare_auto_confirms_when_b_is_unique():
    """A≥1件・B=1件（表2 左上セル）: 自動確定して比較結果を返し、
    `assembly_compare_b` にBのファイル名を格納する。"""
    results = {
        'a.dxf': {'rows': [{'機器符号候補': 'Y', 'ラベル': 'R10', '個数': 1}],
                  'title': 'UNIT内結線図', 'main_drawing_number': 'EE1'},
        'b.dxf': {'rows': [{'機器符号候補': 'Y', 'ラベル': 'R10', '個数': 1}],
                  'title': '組立図', 'main_drawing_number': 'EE2'},
    }
    at = _run_resolve(results)
    assert at.exception.len == 0
    result = at.session_state['_test_result']
    assert result['b_file'] == 'b.dxf'
    assert at.session_state['assembly_compare_b'] == 'b.dxf'
    assert 'assembly_compare_candidates' not in at.session_state
    assert 'assembly_compare_warning' not in at.session_state


def test_resolve_assembly_compare_warns_when_no_wiring_files():
    """A=0件（表2 下段、D4）: 警告文言をセットし None を返す。
    B側の判定は行わないため候補キーもセットされない。"""
    results = {
        'a.dxf': {'rows': [], 'title': 'ＳＹＳＴＥＭ＿機器銘板'},
        'b.dxf': {'rows': [], 'title': '組立図'},
    }
    at = _run_resolve(results)
    assert at.exception.len == 0
    assert at.session_state['_test_result'] is None
    assert 'UNIT内結線図' in at.session_state['assembly_compare_warning']
    assert 'assembly_compare_candidates' not in at.session_state


def test_resolve_assembly_compare_sets_candidates_when_b_has_multiple_matches():
    """A≥1件・B=複数件（表2 右上セル）: 自動確定せず、候補一覧を
    `assembly_compare_candidates` に格納して None を返す。"""
    results = {
        'a.dxf': {'rows': [], 'title': 'UNIT内結線図'},
        'b1.dxf': {'rows': [], 'title': '組立図（上部）'},
        'b2.dxf': {'rows': [], 'title': '組立図（下部）'},
    }
    at = _run_resolve(results)
    assert at.exception.len == 0
    assert at.session_state['_test_result'] is None
    assert at.session_state['assembly_compare_candidates'] == ['b1.dxf', 'b2.dxf']
    assert 'assembly_compare_b' not in at.session_state


def test_resolve_assembly_compare_falls_back_to_all_files_when_b_has_no_matches():
    """A≥1件・B=0件（表2 中央上セル）: 自動確定せず、全ファイル名を候補に
    格納して None を返す（ユーザーが手動でBを選べるようにするため）。"""
    results = {
        'a.dxf': {'rows': [], 'title': 'UNIT内結線図'},
        'other.dxf': {'rows': [], 'title': 'ＳＹＳＴＥＭ＿機器銘板'},
    }
    at = _run_resolve(results)
    assert at.exception.len == 0
    assert at.session_state['_test_result'] is None
    assert at.session_state['assembly_compare_candidates'] == ['a.dxf', 'other.dxf']
