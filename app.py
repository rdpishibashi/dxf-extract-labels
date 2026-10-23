import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import time
import traceback
from io import BytesIO
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model')
sys.path.insert(0, model_path)

from model.region_detector import (
    collect_unique_region_candidate_names, resolve_globally_decided_name,
    pending_candidates_for_region,
)
from model.ref_designator import unnamed_region_labels
from model.excel_output import (
    create_ref_designator_excel_output, create_region_excel_output,
    display_width, autofit_width, DRAWING_LIST_COLUMN_WIDTH,
)
from model import ref_designator
from model import region_preview
from model import terminal_detector
from model import assembly_compare
from model import designator_master
from config import region_detection_config
from view import region_selection


def handle_error(e, show_traceback=True):
    """エラーを適切に処理して表示する"""
    st.error(f"エラーが発生しました: {str(e)}")
    if show_traceback:
        st.error(traceback.format_exc())


def _resolve_master_index():
    """機器符号一覧の索引（`designator_master.DesignatorMasterIndex`）を解決する。

    鍵の解決順は `st.secrets.get('DESIGNATOR_MASTER_KEY')` → 環境変数
    `DESIGNATOR_MASTER_KEY`（`st.secrets` はローカルの `.streamlit/secrets.toml`
    とStreamlit Cloudの Secrets の両方を自動で読むため、これだけで両環境を
    カバーできる。ファイルが無い等で例外を投げる場合があるため try/except で
    包む）。

    鍵が無い・ダイジェストファイルが無い・鍵が一覧生成時と異なる（鍵不一致）
    のいずれでも索引は `None` になり、機器符号判断はCAN/空欄のみの従来動作に
    なる（2026-10-08ユーザー決定）。呼び出し元は本関数の2番目の戻り値
    （状態メッセージ）を画面に表示し、どの状態かが分かるようにする
    （「利用可のときも件数を出す」——設定済みのつもりで実は無効、という
    逆方向の事故も防ぐため、鍵が無い場合だけでなく読み込めた場合も表示する）。

    戻り値: (master_index または None, 状態メッセージ文字列)。
    """
    try:
        secret = st.secrets.get('DESIGNATOR_MASTER_KEY')
    except Exception:
        secret = None
    if not secret:
        secret = os.environ.get('DESIGNATOR_MASTER_KEY')

    digest_path = os.path.join(current_dir, 'assets', 'designator_digests.txt')

    if not secret:
        return None, ("機器符号一覧が未設定のため DEF(defined) 判定は行われていません")
    if not os.path.isfile(digest_path):
        return None, ("機器符号一覧のダイジェストファイルが見つからないため "
                       "DEF(defined) 判定は行われていません")

    index = designator_master.load_index(digest_path, secret)
    if index is None:
        return None, ("鍵が機器符号一覧の生成時と異なるため DEF(defined) 判定は行われていません")
    return index, f"機器符号一覧: {len(index)}件を読み込み済み（DEF判定あり）"

st.set_page_config(
    page_title="DXF Extract Labels",
    page_icon="📝",
    layout="wide",
)

# ============================================================
# session_state クリア対象キー（3つのボタンハンドラで使用）
#
# 3つのリストは意図的にそれぞれ異なる範囲を持つ（同一のベースから機械的に
# 導出できるわけではない）:
#   - REGION_DETECT: 「領域を検出」再実行時。
#   - EXTRACT: 「ラベルを抽出」実行時。入力状態（region_analyses・uploaded
#     files 等）は保持したまま、前回の抽出結果のみクリアする。
#   - RESTART: 「新しい抽出を開始」時。入力状態を含む全リセット
#     （オプション設定ウィジェットの値は対象外＝保持される、ユーザー指定）。
# ============================================================

# `analyze_dxf_regions()` が図面枠を検出できなかった場合に設定するエラー文言の
# 一部（`model/region_detector.py` 参照。frame_lineweight の値を跨いで安定する
# 部分文字列のみを使う）。図面枠がPaper Spaceにのみ存在する等の既知の未対応
# ケースで、現時点では対応予定が無いため「領域の確認」セクションに表示しない
# （2026-09-11 ユーザー指示）。ラベル抽出（機器符号抽出）自体は既存のフォール
# バックで正常動作するため、この抑制は「領域の確認」UIのみに限定する
# （Excel出力・他セクションには影響しない）。
_FRAME_NOT_FOUND_ERROR_MARKER = region_selection.FRAME_NOT_FOUND_ERROR_MARKER

# 「結線図-組立図を比較」用のセッションキー（v3.7.0、「図面種別判定結果」
# セクション）。抽出結果本体（`extract_results_full`/`extract_sort_value`）は
# 「判定を実行」時のExcel再生成に、`type_judgment`（現在のチェック状態＝正本）
# /`type_judgment_applied`（最後にExcelへ反映した判定。Noneは未反映）は
# セクションの表示・ボタン色に使う。3つのクリア対象リストすべてに含める
# （いずれも「前回の抽出結果が古くなるタイミング」で揃って無効化すべきため）。
# `type_judgment_ver`（data_editorのkeyバージョン）は含めない——ここで
# クリアして0に戻すと、ブラウザ側に残っている古いバージョンのwidgetと
# keyが衝突しうるため、セッションを通じて増え続けるだけにする。
_ASSEMBLY_COMPARE_CLEAR_KEYS = [
    'extract_results_full', 'extract_sort_value',
    'type_judgment', 'type_judgment_applied',
]

_REGION_DETECT_CLEAR_KEYS = [
    'excel_result', 'is_region_mode', 'region_results_summary',
    'region_selection_confirmed',
] + _ASSEMBLY_COMPARE_CLEAR_KEYS
_REGION_DETECT_CLEAR_PREFIXES = ('rc_', 'grc_', 'gre_', 'regprev_')

_EXTRACT_CLEAR_KEYS = [
    'excel_result', 'is_region_mode', 'region_results_summary',
    'ref_results_summary', 'terminal_results', 'extract_preview_expanded',
] + _ASSEMBLY_COMPARE_CLEAR_KEYS
_EXTRACT_CLEAR_PREFIXES = ('regprev_result_',)

_RESTART_CLEAR_KEYS = [
    'excel_result', 'output_filename', 'is_region_mode', 'region_analyses',
    'region_results_summary', 'ref_results_summary', 'download_done',
    'terminal_results', 'region_selection_confirmed', 'extract_preview_expanded',
] + _ASSEMBLY_COMPARE_CLEAR_KEYS
_RESTART_CLEAR_PREFIXES = ('rc_', 'grc_', 'gre_', 'regprev_')


_temp_dxf_file = region_selection.temp_dxf_file

def _clear_session_keys(keys=(), prefixes=()):
    """指定した session_state キー（完全一致）とキー接頭辞（前方一致）をまとめて
    削除する（「領域を検出」「ラベルを抽出」「新しい抽出を開始」各ハンドラの
    前回結果クリア処理で共用）。存在しないキーは無視する。"""
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    if prefixes:
        for k in list(st.session_state.keys()):
            if isinstance(k, str) and k.startswith(prefixes):
                del st.session_state[k]


def _compute_terminal_results(uploaded_files, status_placeholder=None, start_time=None):
    """「端子一覧を抽出」オプション用に、アップロード済み全ファイルの
    端子台(TB)矩形解析を実行する（`terminal_detector.analyze_dxf_terminals`）。
    対象外ファイル（タイトルが「UNIT内結線図」でない等）は
    `is_target=False` のまま結果に含まれる（Excel出力側で除外される）。

    `status_placeholder`/`start_time` を渡すと、DXF-diff-managerと同じ
    書式（ファイル名＋件数＋経過秒数）で処理中のファイル名を表示する
    （2026-09-23 ユーザー指示。省略時は表示しない）。"""
    results = {}
    total = len(uploaded_files)
    for idx, uf in enumerate(uploaded_files, start=1):
        if status_placeholder is not None:
            elapsed = time.time() - start_time
            status_placeholder.text(
                f"端子台を解析中: {uf.name}（{idx}/{total}件、経過 {elapsed:.1f} 秒）")
        with _temp_dxf_file(uf) as tmp:
            results[uf.name] = terminal_detector.analyze_dxf_terminals(
                tmp, original_filename=uf.name)
    return results


# 領域選択UIの本体は共有モジュール `view/region_selection.py`（DXF-label-comparison と
# バイト一致で共有）に移した（v3.8.0）。既存の呼び出し・テストのため旧名を残す。
_global_checked_region_names = region_selection.global_checked_region_names
_gather_name_selections = region_selection.gather_name_selections
_on_change_radio = region_selection.on_change_radio
_on_change_none = region_selection.on_change_none
_compute_default_candidate_index = region_selection.compute_default_candidate_index
_render_corners_popover = region_selection.render_corners_popover
_render_dangling_edges_section = region_selection.render_dangling_edges_section


def _render_intro():
    """タイトル・フォント調整CSS・「プログラム説明」パネルを表示する。"""
    st.title('DXF Extract Labels - 機器符号抽出')

    # 日本語の文字だけを英数字より小さく（94%）表示する。
    # @font-face の unicode-range で日本語グリフ範囲だけ別フォント定義にし、
    # size-adjust で縮小率を指定する（文字単位で自動的に使い分けられるため、
    # ウィジェットごとのフォントサイズ指定は不要）。st.dataframe/Plotly の
    # Canvas 描画部分には効かない場合がある。
    st.markdown("""
        <style>
        /* 日本語グリフ専用のフォント定義: 94%縮小（ユーザー調整済み） */
        @font-face {
            font-family: "AppMixedFont";
            src: local("Hiragino Kaku Gothic ProN"), local("Yu Gothic UI"),
                 local("Yu Gothic"), local("Meiryo");
            unicode-range: U+3000-303F,  /* 句読点・記号 */
                           U+3040-30FF,  /* ひらがな・カタカナ */
                           U+FF00-FFEF,  /* 全角英数・半角カナ */
                           U+4E00-9FFF, U+3400-4DBF;  /* 漢字 */
            size-adjust: 94%;
        }
        /* 上記範囲外（英数字）は通常サイズのフォントにフォールバック */
        @font-face {
            font-family: "AppMixedFont";
            src: local("Source Sans Pro"), local("Helvetica Neue"), local("Arial");
        }
        .stApp, .stApp p, .stApp li, .stApp label, .stApp td, .stApp th,
        .stApp h1, .stApp h2, .stApp h3, .stApp input, .stApp button {
            font-family: "AppMixedFont", sans-serif !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.write('DXFファイルからテキストラベルを抽出し、Excel形式で出力します。')

    with st.expander("ℹ️ プログラム説明", expanded=False):
        st.info("\n".join([
            "このツールは、DXFファイルからテキスト要素（ラベル）を抽出し、Excelファイルに出力します。"
            "図面枠内・図面情報枠（タイトルブロック）外の全ラベルを抽出し、"
            "機器符号（Reference Designator）パターンに一致するかどうかを"
            "各ラベルに「機器符号候補」として付記します。図番・タイトル・"
            "サブタイトルも常にSummaryシートに記録します。",
            "",
            "**主なオプション：**",
            "- 領域を検出：図面枠内の矩形領域を検出し、領域ごとにラベルを分類します",
            "- 端子一覧を抽出：「UNIT内結線図」の図面を対象に、端子台の端子番号を"
            "「TB List」シートに出力します",
            "- 結線図-組立図を比較：「UNIT内結線図」の図面（A）と「組立図」の図面（B）の"
            "ラベルを比較し、「結線図-組立図比較」シートに出力します。Bが1件に"
            "自動特定できない場合は、抽出後に比較する図面を選択できます",
            "",
            "**使用手順：**",
            "1. DXFファイルをアップロードしてください（複数可）",
            "2. 必要に応じてオプションを設定します",
            "3. 「領域を検出」をONにした場合は、「領域を検出」ボタンで矩形領域を検出し、"
            "「抽出領域の選択」で抽出したい領域名にチェックを入れます（複数選択可）。"
            "チェックで決定できなかった領域は「領域の確認」で個別に選択します",
            "4. 「ラベルを抽出」ボタンで処理を実行します（結果はそのままExcelとして"
            "ダウンロードできます）。抽出後はオプション等を振り返れますが変更は"
            "できません（変更するには「新しい抽出を開始」を押してください）",
            "",
            "**Excelファイルの内容：**",
            "- Summaryシート：全ファイルの抽出結果概要（総抽出ラベル数・機器符号候補"
            "ラベル数・ページ数・検出領域数・図番・タイトル・サブタイトル）",
            "- 抽出ラベル一覧シート：全ファイル横断のラベル集計（領域検出時も出力されます）",
            "- 領域一覧・領域別ラベル一覧シート（領域検出時）：検出した領域ごとの集計"
            "（領域一覧は検出結果にかかわらず検出した全領域を記載します）",
            "- 各ファイルシート：個別ファイルの抽出ラベル一覧（機器符号候補列付き、"
            "領域検出時は「領域」列も付きます）",
            "- TB Listシート（端子一覧を抽出時）：端子台ごとの端子番号一覧",
            "- 結線図-組立図比較シート（結線図-組立図を比較時）：A・Bのラベル比較"
            "（機器符号候補・ラベル・A合計出現数・A図番数・B出現数・A図番・比較結果）",
            "- Summary以外の全シートにはオートフィルタが設定されています",
        ]))


def _render_upload_section():
    """ファイルアップロードUIを表示し、アップロードされたファイルのリストを返す
    （空/Noneの場合は呼び出し元が早期returnする）。"""
    # ============================================================
    # ファイルアップロード
    # ============================================================
    st.subheader("DXFファイルのアップロード")
    if 'uploader_version' not in st.session_state:
        st.session_state['uploader_version'] = 0
    uploaded_files = st.file_uploader(
        "DXFファイルを選択してください（複数可，フォルダ可）",
        type="dxf",
        accept_multiple_files=True,
        key=f"dxf_uploader_{st.session_state['uploader_version']}",
    )
    return uploaded_files


def _render_options_section(locked):
    """オプション欄（領域検出・端子一覧抽出・結線図-組立図比較・並び替え・
    出力ファイル名）を表示し、選択値を dict で返す。

    v3.0.0で「機器符号（候補）以外も抽出」「図面番号・タイトル・サブタイトルを
    抽出」の2オプションを廃止した。図面枠内・図面情報枠外の全ラベルを常に抽出し
    （機器符号候補かどうかはラベルごとの列として出力）、図番・タイトル・
    サブタイトルも常に抽出する。

    `locked` が True（抽出結果表示中）の間は全ウィジェットを disabled にし、
    選択内容を振り返れるが変更できない状態にする。"""
    # ============================================================
    # オプション
    # ============================================================
    st.subheader("オプション")
    col_left, col_right = st.columns(2)

    with col_left:
        # a) 領域を検出（矩形領域抽出機能全体のON/OFF）
        enable_region_detection = st.checkbox(
            "領域を検出",
            value=False,
            key='enable_region_detection',
            disabled=locked,
            help="ONにすると「領域を検出」ボタン、「抽出領域の選択」と「領域の確認」"
                 "セクションが表示され、矩形領域付きでラベルを抽出できます。"
        )
        if not enable_region_detection:
            # OFF中に前回検出済みの領域データ（region_analyses）が残っていると、抽出時に
            # 誤って領域付きモードと判定されてしまうため、表示を隠すだけでなく
            # session_state からも明示的に消しておく。
            # このブロックは「領域を検出」OFF中は毎回の再描画で実行されるため、
            # _REGION_DETECT_CLEAR_KEYS（excel_result 等の抽出結果本体）を含めては
            # ならない（含めると、通常モードで生成した抽出結果が直後の再描画で
            # 消えてしまう。2026-07-15ユーザー報告で発覚、v1.9.4で領域検出を
            # オプション化した際に混入した回帰）。
            _clear_session_keys(
                keys=['region_analyses', 'region_selection_confirmed'],
                prefixes=_REGION_DETECT_CLEAR_PREFIXES)

        # b) 端子一覧を抽出
        extract_terminal_option = st.checkbox(
            "端子一覧を抽出",
            value=False,
            disabled=locked,
            help="タイトルが「UNIT内結線図」の図面を対象に、TBで始まる"
                 "端子台ラベルに対応する矩形構成を検出し、矩形内の"
                 "端子番号を「TB List」シートに出力します。"
                 "「端子台」ごとに同じ番号が複数ある場合は件数をカッコで"
                 "表示します（例: 7(2)）。\n\n"
                 "「TB COMPONENT」の図面（端子台の全端子番号一覧ページ）"
                 "は対象外です。"
        )

        # c) 結線図-組立図を比較
        compare_assembly_option = st.checkbox(
            "結線図-組立図を比較",
            value=False,
            disabled=locked,
            help="タイトルが「UNIT内結線図」の図面（複数可）を結線図、"
                 "タイトルに「組立図」を含む図面（1件）を組立図として、両者の"
                 "ラベル（機器符号）を比較し「結線図-組立図比較」シート"
                 "（抽出ラベル一覧シート直前）に出力します。\n\n"
                 "抽出後に表示される「図面種別判定結果」セクションで、どの"
                 "図面を結線図・組立図としたかを確認・修正できます。自動判定"
                 "が確定できない場合（組立図が0件・複数件、結線図が0件等）は、"
                 "比較シート無しでラベル抽出処理は進めます。"
        )

    with col_right:
        sort_option = st.selectbox(
            "並び替え",
            options=[
                ("昇順", "asc"),
                ("降順", "desc"),
                ("並び替えなし", "none")
            ],
            format_func=lambda x: x[0],
            help="ラベルの並び替え順を指定します",
            index=0,
            disabled=locked,
        )
        sort_value = sort_option[1]

        output_filename = st.text_input(
            "出力Excelファイル名",
            value="extracted_labels.xlsx",
            help="出力するExcelファイルの名前を指定します",
            disabled=locked,
        )
        if not output_filename.endswith('.xlsx'):
            output_filename += '.xlsx'
    return {
        'enable_region_detection': enable_region_detection,
        'extract_terminal_option': extract_terminal_option,
        'compare_assembly_option': compare_assembly_option,
        'sort_value': sort_value,
        'output_filename': output_filename,
    }


def _build_region_cfg():
    """`config.py` の `region_detection_config` から `region_cfg` を組み立てる
    （本体は `region_selection.build_region_cfg()`。2026-09-10、UIフォームを撤去し
    config.py 編集方式に統一した）。既定値のみの場合、返り値は
    `DEFAULT_REGION_CONFIG` と完全に等価になる（`tests/unit/test_config_region_cfg.py`
    で固定）。"""
    return region_selection.build_region_cfg(region_detection_config)


# st.spinner は開始から 0.5 秒経ってから表示される（streamlit/elements/spinner.py の
# DELAY_SECS）。その前に進捗用の st.empty() を作ると進捗メッセージが先に出て、
# spinner が出た時点でその下へ動いてしまう。spinner の表示後に作るため少し待つ。
SPINNER_APPEAR_WAIT_SEC = 0.6


def _status_placeholder_after_spinner():
    """spinner が表示されるのを待ってから進捗表示用の `st.empty()` を返す。"""
    time.sleep(SPINNER_APPEAR_WAIT_SEC)
    return st.empty()


def _render_region_detection_trigger(uploaded_files, region_cfg):
    """「領域を検出」ボタンを表示し、押されたら各ファイルを解析して
    `st.session_state['region_analyses']` に格納する。

    表示するかどうか（領域検出ON かつ未検出時のみ）は呼び出し元の `app()` が
    判定する。表示される場面は常に「これが次に行うべき唯一の操作」であるため、
    常に primary（青）で表示する。解析本体は `region_selection.run_region_detection()`
    （ファイルキー＝アップロードファイル名）。"""
    # ============================================================
    # 領域を検出
    # ============================================================
    if st.button("領域を検出", key="detect_regions_btn", type="primary"):
        try:
            with st.spinner('領域を検出中...'):
                # 進捗表示（ファイル名・件数・経過秒数）は spinner の下に出す
                # ため、st.empty() は spinner の内側（2026-09-23 ユーザー指示、
                # DXF-diff-managerと同じ書式。`_handle_extract_button()` 参照）。
                status_placeholder = _status_placeholder_after_spinner()
                # 面積条件に満たない小領域も検出しておく（採用は「ラベルを抽出」時に、
                # 全ファイルで採用された名称と一致するものだけ。`ref_designator.
                # add_minor_region_selections()`）。他プロジェクトは検出しない。
                analyses = region_selection.run_region_detection(
                    [(uf.name, uf) for uf in uploaded_files],
                    {**region_cfg, 'detect_minor_regions': True},
                    status_placeholder, time.time())
                status_placeholder.empty()
                st.session_state['region_analyses'] = analyses
                _clear_session_keys(
                    keys=_REGION_DETECT_CLEAR_KEYS,
                    prefixes=_REGION_DETECT_CLEAR_PREFIXES)
            st.rerun()
        except Exception as e:
            handle_error(e)


_invalidate_region_selection_confirmation = \
    region_selection.invalidate_region_selection_confirmation
_region_confirmation_targets = region_selection.region_confirmation_targets


def _render_global_region_names_section(locked):
    """「領域一覧」セクション（本体は `region_selection.render_global_region_names_section()`）。
    検出済みの全ファイル・全領域の名称候補を重複なくABC順で一覧表示する。
    未検出なら何も表示しない。"""
    if 'region_analyses' not in st.session_state:
        return
    analyses = st.session_state['region_analyses']
    # 名称のない領域も「領域一覧」に出す（表示名は ref_designator.unnamed_region_label()。
    # 「特定」は既定ON。同じ表示名は1行にまとめる）。
    unnamed = list(dict.fromkeys(
        label for analysis in analyses.values()
        for label in unnamed_region_labels(analysis).values()))
    # 5%未満の小領域の採用オプション（抽出時にだけ効くため `region_selection_confirmed` は
    # 戻さない）。「領域一覧」の表の右に置く。
    def _minor_option():
        st.checkbox(MINOR_INCLUDE_CONFIRMED_LABEL, key=MINOR_INCLUDE_CONFIRMED_KEY,
                    disabled=locked, help=MINOR_INCLUDE_CONFIRMED_HELP)

    region_selection.render_global_region_names_section(
        collect_unique_region_candidate_names(analyses) + unnamed, locked,
        allow_exclude=True, default_on_names=unnamed, side_widget=_minor_option,
        show_select_all=True)


REGION_CONFIRM_HEIGHT_PX = 600
REGION_PREVIEW_HEIGHT_PX = 720
REGION_PREVIEW_WARN_BYTES = 8 * 1024 * 1024


@st.cache_data(show_spinner=False, max_entries=20)
def _render_region_base(dxf_bytes: bytes, version: int = region_preview.BASE_RENDER_VERSION):
    """元の図面全体の白 SVG（DXFバイト列と `BaseRender` の版をキーにキャッシュ）。
    `BaseRender` の項目を変えたら版を上げる（古い形のキャッシュが残って AttributeError に
    なるのを防ぐ。実行中のアプリはコードを更新してもキャッシュが残るため）。"""
    return region_preview.render_base_svg(dxf_bytes)


# 「領域の確認」の双方向 viewer（選択を Streamlit へ返す。モジュールで1回だけ登録する）
_REGION_CONFIRM_VIEWER = st.components.v2.component(
    "region_confirm_viewer",
    html=region_preview.CONFIRM_HTML,
    css=region_preview.CONFIRM_CSS,
    js=region_preview.CONFIRM_JS,
)


def _unnamed_regions(analysis):
    """名称の無い領域 `[(reg, 表示名, 特定ON?), ...]`（「抽出領域の選択」の `no name N` と同じ表示名）。"""
    labels = unnamed_region_labels(analysis)
    by_id = {r['id']: r for r in analysis.get('regions', [])}
    return [(by_id[rid], label, bool(st.session_state.get(f"grc_{label}", True)))
            for rid, label in labels.items() if rid in by_id]


def _decided_regions(analysis, pending_regions):
    """要選択ではないが名称が確定している領域 `[(reg, 名称), ...]`（viewer の赤表示用）。"""
    checked = region_selection.global_checked_region_names()
    excluded = region_selection.global_excluded_region_names()
    pending_ids = {r['id'] for r in pending_regions}
    out = []
    for reg in analysis.get('regions', []):
        if reg['id'] in pending_ids or region_selection._region_excluded(reg, excluded):
            continue
        name = resolve_globally_decided_name(reg, checked)
        if name is None:
            name = region_selection.resolve_by_other_frames(reg, analysis, checked, excluded)
        if name is not None:
            out.append((reg, name))
    return out


def _show_region_confirm_viewer(fname, dxf_bytes, analysis, pending_regions, locked):
    """「領域の確認」の viewer。viewer 内の選択を正本（`rc_…`）へ反映する。"""
    checked = region_selection.global_checked_region_names()
    try:
        base = _render_region_base(dxf_bytes)
        data = region_preview.build_confirmation_data(
            base, fname, analysis, pending_regions, _decided_regions(analysis, pending_regions),
            st.session_state, checked, locked,
            region_selection.match_typed_region_name, pending_candidates_for_region,
            unnamed=_unnamed_regions(analysis))
        res = _REGION_CONFIRM_VIEWER(
            key=f"regprev_viewer_{fname}", data=data, height=REGION_CONFIRM_HEIGHT_PX,
            on_choice_change=lambda: None)
    except Exception as e:
        st.warning(f"プレビューを表示できませんでした: {e}")
        return
    choice = res.choice if hasattr(res, "choice") else None
    seq_key = f"regprev_seq_{fname}"
    if choice and choice.get('seq') != st.session_state.get(seq_key) and not locked:
        st.session_state[seq_key] = choice.get('seq')
        reg = next((r for r in pending_regions if r['id'] == choice.get('region')), None)
        if reg is not None:
            region_selection.apply_viewer_choice(
                fname, reg, choice.get('kind'), checked, idx=choice.get('idx'),
                text=choice.get('text', ''), typed_text=choice.get('typed_text'))
            st.rerun()


def _show_region_viewer(dxf_bytes, overlays, height_px):
    """領域ビューアを描画する。失敗してもアプリ全体は止めない。"""
    try:
        base = _render_region_base(dxf_bytes)
        html_text = region_preview.build_region_viewer_html(base, overlays, height_px)
        if not base.cjk_font and not base.empty:
            st.caption("⚠️ 日本語フォントが見つからないため、日本語の文字はプレビューに表示されません。")
        if len(html_text) > REGION_PREVIEW_WARN_BYTES:
            st.caption(f"⚠️ 図面が大きいため表示に時間がかかる場合があります（約 {len(html_text) / 1024 / 1024:.0f}MB）。")
        components.html(html_text, height=height_px)
    except Exception as e:
        st.warning(f"プレビューを表示できませんでした: {e}")


def _render_region_confirmation_section(enable_region_detection, locked, uploaded_files=()):
    """「領域の確認」セクション（本体は `region_selection.render_region_confirmation_section()`）。
    「領域選択を完了」が押される（`region_selection_confirmed` が True になる）
    までは表示しない（2026-07-30 ユーザー指定）。"""
    # ============================================================
    # 領域の確認
    # ============================================================
    if not (enable_region_detection and 'region_analyses' in st.session_state
            and st.session_state.get('region_selection_confirmed')):
        return
    files = {uf.name: uf for uf in uploaded_files}

    def _file_preview(fname, analysis, pending_regions):
        """回路全体（白）の中で、選択が必要な領域を赤でハイライトする。"""
        uf = files.get(fname)
        if uf is None or not (pending_regions or unnamed_region_labels(analysis)):
            return False
        if not st.toggle("図面で領域を確認", value=False, key=f"regprev_confirm_{fname}"):
            return False  # OFF: ページ上に選択肢を出す（選択状態は viewer と共通の正本）
        _show_region_confirm_viewer(fname, uf.getvalue(), analysis, pending_regions, locked)
        return True  # ON: 選択は viewer の中で行う

    region_selection.render_region_confirmation_section(
        st.session_state['region_analyses'], locked, file_preview=_file_preview,
        allow_typed_name=True, show_unnamed=True)


def _resolve_assembly_compare(results, compare_assembly_option, master_index=None):
    """「結線図-組立図を比較」ON時、抽出結果からA（結線図）・B（組立図）を
    タイトルで自動判定する（`assembly_compare.judge_drawing_types()`）。

    自動判定がそのまま成立する場合（結線図1件以上・組立図ちょうど1件・
    重複なし、`assembly_compare.validate_type_selection()` 参照）はその場で
    比較結果を返し、`type_judgment_applied` に判定内容を記録する。成立
    しない場合は比較結果 None を返す（比較シート無しでExcelを生成する。
    ラベル抽出自体は成立可否に関わらず常に実行される）——「図面種別判定結果」
    セクション（`_render_type_judgment_section()`）でユーザーが確認・修正
    したうえで「判定を実行」する（v3.7.0。旧・自動確定/候補selectbox方式
    〈D2/D3〉を置き換えた）。

    OFF時は `type_judgment` 系のキーを一切設定しない（前回分は呼び出し元の
    `_EXTRACT_CLEAR_KEYS` 等で既にクリア済みのため、セクション自体が
    非表示になる）。
    """
    if not compare_assembly_option:
        return None

    selection = assembly_compare.judge_drawing_types(results)
    st.session_state['type_judgment'] = selection
    st.session_state['type_judgment_ver'] = st.session_state.get('type_judgment_ver', 0) + 1

    validation = assembly_compare.validate_type_selection(selection)
    if validation['errors']:
        st.session_state['type_judgment_applied'] = None
        return None

    st.session_state['type_judgment_applied'] = dict(selection)
    return assembly_compare.compare_assembly(
        results, validation['wiring_files'], validation['b_fname'], master_index=master_index)


def _handle_extract_button(uploaded_files, region_cfg, opts, master_index=None):
    """「ラベルを抽出」ボタンを表示し、押されたらモード（領域付き/通常）に
    応じた抽出処理を実行する。

    表示するかどうか（結果表示中でない、かつ領域検出OFFまたは検出済み）は
    呼び出し元の `app()` が判定する。表示される場面は常に「これが次に
    行うべき唯一の操作」であるため、常に primary（青）で表示する。

    v3.0.0で「機器符号（候補）以外も抽出」「図面番号・タイトル・サブタイトルを
    抽出」の2オプションを廃止したため、モードは領域付き/通常の2つのみ
    （旧4分岐から単純化）。図面枠内・図面情報枠外の全ラベルを常に抽出し
    （機器符号候補かどうかはラベルごとの列として出力）、図番・タイトル・
    サブタイトルも常に抽出する。ファイルごとの抽出本体は共有モジュールの
    `region_selection.run_label_extraction()`（→ `model/extraction_pipeline.py`）。"""
    extract_terminal_option = opts['extract_terminal_option']
    compare_assembly_option = opts['compare_assembly_option']
    sort_value = opts['sort_value']
    output_filename = opts['output_filename']

    # ============================================================
    # ラベルを抽出
    # ============================================================
    if st.button("ラベルを抽出", type="primary"):
        try:
            frame_lineweight = int(region_cfg['frame_lineweight'])
            with st.spinner(f'{len(uploaded_files)}個のDXFファイルを処理中...'):
                # 進捗表示（ファイル名・件数・経過秒数）は spinner の下に出す
                # ため、st.empty() は spinner の内側（2026-09-23 ユーザー指示）。
                status_placeholder = _status_placeholder_after_spinner()
                start_time = time.time()

                # 前回の結果をクリアする
                _clear_session_keys(
                    keys=_EXTRACT_CLEAR_KEYS,
                    prefixes=_EXTRACT_CLEAR_PREFIXES)

                if extract_terminal_option:
                    st.session_state['terminal_results'] = _compute_terminal_results(
                        uploaded_files, status_placeholder, start_time)
                terminal_results = st.session_state.get('terminal_results')

                if 'region_analyses' in st.session_state:
                    # 領域付きモード: 抽出して直ちにExcelを生成する
                    analyses = st.session_state['region_analyses']
                    ref_data_by_file = region_selection.run_label_extraction(
                        [(uf.name, uf) for uf in uploaded_files], frame_lineweight,
                        analyses=analyses, status_placeholder=status_placeholder,
                        start_time=start_time)

                    name_selections_by_file = {
                        fname: _gather_name_selections(fname, analyses[fname])
                        for fname in ref_data_by_file
                    }
                    minor_targets = (
                        ref_designator.adopted_region_names(name_selections_by_file)
                        if st.session_state.get(MINOR_INCLUDE_CONFIRMED_KEY)
                        else region_selection.unconditional_region_names(analyses))
                    name_selections_by_file = ref_designator.add_minor_region_selections(
                        analyses, name_selections_by_file, minor_targets,
                        excluded_names=region_selection.global_excluded_region_names())
                    region_results = ref_designator.build_ref_designator_region_results(
                        ref_data_by_file, analyses, name_selections_by_file, sort_value,
                        master_index=master_index,
                        excluded_names=region_selection.global_excluded_region_names())
                    assembly_result = _resolve_assembly_compare(
                        region_results, compare_assembly_option, master_index=master_index)
                    st.session_state.excel_result = create_region_excel_output(
                        region_results, terminal_results=terminal_results,
                        assembly_compare=assembly_result, master_index=master_index)
                    st.session_state.output_filename = output_filename
                    st.session_state['is_region_mode'] = True
                    st.session_state['region_results_summary'] = {
                        f: {k: v for k, v in d.items() if k not in ('named', 'rows', 'region_rows')}
                        for f, d in region_results.items()
                    }
                    st.session_state['extract_results_full'] = region_results
                    st.session_state['extract_sort_value'] = sort_value

                else:
                    # 通常モード: 抽出して直ちにExcelを生成する
                    ref_data_by_file = region_selection.run_label_extraction(
                        [(uf.name, uf) for uf in uploaded_files], frame_lineweight,
                        status_placeholder=status_placeholder, start_time=start_time)

                    ref_final = ref_designator.build_ref_designator_final(
                        ref_data_by_file, sort_value, master_index=master_index)
                    assembly_result = _resolve_assembly_compare(
                        ref_final, compare_assembly_option, master_index=master_index)
                    st.session_state.excel_result = create_ref_designator_excel_output(
                        ref_final, sort_value, terminal_results=terminal_results,
                        assembly_compare=assembly_result, master_index=master_index)
                    st.session_state.output_filename = output_filename
                    st.session_state['is_region_mode'] = False
                    st.session_state['ref_results_summary'] = ref_final
                    st.session_state['extract_results_full'] = ref_final
                    st.session_state['extract_sort_value'] = sort_value

                st.session_state['download_done'] = False
                status_placeholder.empty()

            st.rerun()
        except Exception as e:
            handle_error(e)


def _apply_type_judgment(selection, validation, master_index=None):
    """「判定を実行」ボタンの処理本体: 確定した判定（`validation`）で比較を
    実行し、Excelを再生成する（`extract_results_full`/`extract_sort_value` に
    保持した抽出結果一式を使い、`terminal_results`・`output_filename` も前回と
    同じ値を引き継ぐ。表4「呼び出し元×Excel出力関数」参照——初回抽出時と
    同じ引数一式を渡すことを確認済み。v3.10.0で `master_index` も同様に
    初回抽出時〈`_handle_extract_button()`〉と同じ値を渡す）。呼び出し前に
    `validation['errors']` が空であることを呼び出し元が確認していること。"""
    results = st.session_state['extract_results_full']
    compare_result = assembly_compare.compare_assembly(
        results, validation['wiring_files'], validation['b_fname'], master_index=master_index)

    terminal_results = st.session_state.get('terminal_results')
    if st.session_state.get('is_region_mode', False):
        st.session_state.excel_result = create_region_excel_output(
            results, terminal_results=terminal_results, assembly_compare=compare_result,
            master_index=master_index)
    else:
        st.session_state.excel_result = create_ref_designator_excel_output(
            results, st.session_state['extract_sort_value'],
            terminal_results=terminal_results, assembly_compare=compare_result,
            master_index=master_index)

    st.session_state['type_judgment_applied'] = dict(selection)
    st.session_state['download_done'] = False
    st.rerun()


def _render_type_judgment_section(master_index=None):
    """「図面種別判定結果」セクション（v3.7.0）: 「結線図-組立図を比較」
    ON時に抽出したファイルについて、結線図・組立図の判定結果
    （図番・タイトル・サブタイトルとともに）一覧表示し、チェックボックスで
    手動修正できるようにする。「抽出結果」セクションの直前に独立した
    セクションとして表示し、抽出結果表示中（locked）でもこのセクションだけは
    操作できる。

    `type_judgment`（session_state、現在のチェック状態の正本）が無ければ
    何も表示しない（比較OFF、または抽出前）。チェック変更のたびに
    Excelを再生成するのではなく、「判定を実行」ボタンを押したときにのみ
    再生成する（変更後・未反映の間はボタンを青、反映済みなら白にする）。

    data_editorとの連動は「正本dict + キー再生成」方式を使う（streamlitスキル
    §6参照）——表示中のdata_editorのsession_stateを直接書き換えると無限同期
    ループで画面が固まるため、差分検出時は正本（`type_judgment`）を更新して
    キーのバージョンを上げ、別ウィジェットとして再生成する。
    """
    selection = st.session_state.get('type_judgment')
    results = st.session_state.get('extract_results_full')
    if selection is None or not results:
        return

    st.subheader("図面種別判定結果")

    rows = assembly_compare.type_judgment_rows(results, selection)
    df = pd.DataFrame(rows).set_index('ファイル名')

    ver = st.session_state.get('type_judgment_ver', 0)
    edited = st.data_editor(
        df,
        key=f"type_judgment_editor_{ver}",
        column_config={
            '結線図': st.column_config.CheckboxColumn('結線図'),
            '組立図': st.column_config.CheckboxColumn('組立図'),
            '図番': st.column_config.TextColumn('図番', disabled=True),
            'タイトル': st.column_config.TextColumn('タイトル', disabled=True),
            'サブタイトル': st.column_config.TextColumn('サブタイトル', disabled=True),
        },
        hide_index=True,
    )

    # 差分検出 → 正本更新 → キーのバージョンを上げて再生成（無条件に反映すると
    # 無限同期ループになるため、差分が無ければ何もしない）。
    deltas = {}
    for fname in edited.index:
        wiring = bool(edited.loc[fname, '結線図'])
        assembly = bool(edited.loc[fname, '組立図'])
        current = selection.get(fname, {})
        if wiring != bool(current.get('wiring')) or assembly != bool(current.get('assembly')):
            deltas[fname] = {'wiring': wiring, 'assembly': assembly}
    if deltas:
        selection.update(deltas)
        st.session_state['type_judgment_ver'] = ver + 1
        st.rerun()

    validation = assembly_compare.validate_type_selection(selection)
    if validation['errors']:
        reasons = '\n'.join(f'- {e}' for e in validation['errors'])
        st.warning(f"結線図-組立図比較を実行できませんでした。\n{reasons}")

    already_applied = st.session_state.get('type_judgment_applied') == selection
    run_disabled = bool(validation['errors'])
    run_type = "secondary" if (not run_disabled and already_applied) else "primary"
    if st.button("判定を実行", type=run_type, disabled=run_disabled, key='type_judgment_apply'):
        _apply_type_judgment(selection, validation, master_index=master_index)


MINOR_INCLUDE_CONFIRMED_KEY = 'minor_include_confirmed'
MINOR_INCLUDE_CONFIRMED_LABEL = "すべてのファイルに適用する"
MINOR_INCLUDE_CONFIRMED_HELP = (
    "このオプションを有効にすると（チェックすると）、「特定」した領域名は、アップロードした"
    "すべてのファイルに対して有効な領域となります。デフォルトのチェックしない状態では、"
    "ファイルごとに有効な領域（図面面積の5％以上の領域）以外は抽出対象になりません。")

PX_PER_CHAR = 8       # 表の列幅 px = 表示幅（全角=2）×8 + 余白（Excelの文字数単位から換算）
PX_PADDING = 28


def _chars_to_px(chars):
    return int(chars * PX_PER_CHAR + PX_PADDING)


def sheet_select_width(names):
    """シート選択 selectbox の幅 px（最長のシート名に合わせる。既定の全幅だと
    長すぎてメニューと認識されない、2026-10-17 ユーザー指摘）。"""
    return max(160, _chars_to_px(max((display_width(n) for n in names), default=0)) + 36)


def preview_column_config(df):
    """プレビュー表の列設定 `({列名: Column}, 表の幅px)` を返す。各列はデータ長に合わせ
    （Excel出力と同じ `autofit_width`）、カンマ区切りで図番を複数並べる列は固定幅にする
    （長さに合わせると見づらくなるため）。"""
    config = {}
    total = 0
    for col in df.columns:
        values = df[col].tolist()
        if '図番' in str(col) and any(',' in str(v) for v in values if v is not None and v == v):
            chars = DRAWING_LIST_COLUMN_WIDTH
        else:
            chars = autofit_width(str(col), values)
        px = _chars_to_px(chars)
        config[col] = st.column_config.Column(width=px)
        total += px
    return config, total


CANDIDATE_COLUMN = '機器符号候補'
CANDIDATE_BLANK = '（空欄）'
_CANDIDATE_ORDER = ('DEF', 'CAN')


def candidate_filter_options(series):
    """「機器符号候補」列に実際にある値を `DEF` → `CAN` → `（空欄）` の順で返す
    （画面の絞り込み選択欄用。その他の値があれば最後に追加）。"""
    present = {CANDIDATE_BLANK if (pd.isna(v) or str(v).strip() == '') else str(v)
               for v in series}
    ordered = [v for v in _CANDIDATE_ORDER if v in present]
    ordered += sorted(v for v in present if v not in _CANDIDATE_ORDER and v != CANDIDATE_BLANK)
    if CANDIDATE_BLANK in present:
        ordered.append(CANDIDATE_BLANK)
    return ordered


def filter_by_candidate(df, selected):
    """「機器符号候補」列が `selected`（`candidate_filter_options()` の値）に含まれる行だけを
    返す（index は保持）。表示専用で、Excel出力には使わない。"""
    keys = df[CANDIDATE_COLUMN].apply(
        lambda v: CANDIDATE_BLANK if (pd.isna(v) or str(v).strip() == '') else str(v))
    return df[keys.isin(set(selected))]


def render_candidate_filter(df, key):
    """表の上に「機器符号候補」の絞り込み選択欄を出し、絞り込み後の df を返す。
    列が無い、または値が無いときは選択欄を出さず df をそのまま返す。初期状態は全選択。"""
    if CANDIDATE_COLUMN not in df.columns or df.empty:
        return df
    options = candidate_filter_options(df[CANDIDATE_COLUMN])
    # 選択肢の構成が変わったとき（新しい抽出結果など）に古い選択が残らないよう、key に含める
    selected = st.pills(CANDIDATE_COLUMN, options, selection_mode='multi',
                        default=options, key=f"{key}:{'|'.join(options)}")
    return filter_by_candidate(df, selected or [])


def _render_region_preview(uploaded_files):
    """「領域プレビュー」: ファイルごとに、図面上の領域名を選んで表示する（領域モードのみ）。"""
    full = st.session_state.get('extract_results_full') or {}
    analyses = st.session_state.get('region_analyses') or {}
    files = {uf.name: uf for uf in uploaded_files}
    with st.expander("領域プレビュー", expanded=False):
        st.caption("ファイルごとに「表示」を ON にすると、図面全体（白）の上に領域を重ねて表示します。"
                   "表示する領域は図面の左のチェックボックスで選べます。")
        for fname, entry in full.items():
            rows = (entry or {}).get('region_rows') or []
            analysis = analyses.get(fname) or {}
            overlays = region_preview.preview_overlays(analysis, rows) if rows else []
            col_name, col_toggle = st.columns([5, 1])
            if not overlays or fname not in files:
                col_name.write(f"{fname}　領域なし")
                continue
            n_target = sum(1 for o in overlays if o.style == "target")
            col_name.write(f"{fname}　特定 {n_target}／検出 {len(overlays)} 領域")
            if col_toggle.toggle("表示", key=f"regprev_result_{fname}"):
                _show_region_viewer(files[fname].getvalue(), overlays, REGION_PREVIEW_HEIGHT_PX)


def _render_results_section(master_index_status=None, uploaded_files=()):
    """抽出結果（統計・ダウンロードボタン・再抽出ボタン）を表示する。

    v3.0.0で「機器符号（候補）以外も抽出」「図面番号・タイトル・サブタイトルを
    抽出」の2オプションを廃止したため、モードは領域付き/通常の2つのみ
    （旧3分岐から単純化）。「領域の確認」セクションを空のときは表示しなく
    したこと（2026-09-10）に伴い、「抽出結果プレビュー」等の置き場として
    「抽出結果」という見出しのセクションをここに新設した。

    `master_index_status`（v3.10.0新設）: 機器符号一覧（DEF判定）が
    利用できているかどうかの状態メッセージ（`_resolve_master_index()` の
    2番目の戻り値）。利用可・未設定・鍵不一致のいずれでも必ず表示する
    （利用可のときも件数を出すことで、設定済みのつもりで実は無効、という
    逆方向の事故も防ぐ）。

    「結線図-組立図を比較」関連の判定表示・操作（v3.7.0で「図面種別判定結果」
    セクションへ分離）はこの関数の対象外——呼び出し元 `app()` が
    `_render_type_judgment_section()` をこの関数の直前に呼ぶ。"""
    # ============================================================
    # 抽出結果
    # ============================================================
    if st.session_state.get('excel_result'):
        st.subheader("抽出結果")
        if master_index_status:
            st.caption(master_index_status)
        is_region_mode = st.session_state.get('is_region_mode', False)

        if is_region_mode:
            summ = st.session_state.get('region_results_summary', {})
            st.success(f"{len(summ)}個のDXFファイルからラベル抽出が完了しました（領域付き）")
        else:
            summ = st.session_state.get('ref_results_summary', {})
            st.success(f"{len(summ)}個のDXFファイルから機器符号の抽出が完了しました")

        # 「一度開いたら開いたままにする」は、シート選択(selectbox)の変更という
        # 明示的なユーザー操作があった場合のみ反映する（on_change）。st.expander
        # の中身は collapsed 表示中でも毎回実行されるため、ここで無条件に True を
        # 立てると初回表示から常に展開済みになってしまう（DXF-diff-manager の
        # 同型パターン〈diff_preview_expanded〉を踏襲）。
        def _mark_extract_preview_expanded():
            st.session_state['extract_preview_expanded'] = True

        preview_expanded = st.session_state.get('extract_preview_expanded', False)
        with st.expander("抽出結果プレビュー", expanded=preview_expanded):
            preview_xl = pd.ExcelFile(BytesIO(st.session_state.excel_result))
            preview_sheet = st.selectbox(
                "シートを選択",
                preview_xl.sheet_names,
                key="extract_preview_sheet",
                on_change=_mark_extract_preview_expanded,
                width=sheet_select_width(preview_xl.sheet_names),
            )
            preview_df = render_candidate_filter(
                preview_xl.parse(preview_sheet),
                key=f"extract_cand_filter_{preview_sheet}")
            column_config, table_width = preview_column_config(preview_df)
            st.dataframe(preview_df, width=table_width, hide_index=True,
                         column_config=column_config)

        if is_region_mode:
            _render_region_preview(uploaded_files)

        download_done = st.session_state.get('download_done', False)

        st.write(f"出力ファイル：**{st.session_state.output_filename}**")
        downloaded = st.download_button(
            label="抽出結果をダウンロード",
            data=st.session_state.excel_result,
            file_name=st.session_state.output_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="secondary" if download_done else "primary",
        )
        if downloaded and not download_done:
            st.session_state['download_done'] = True
            st.rerun()

        if st.button("🔄 新しい抽出を開始", key="restart_button",
                     type="primary" if download_done else "secondary"):
            _clear_session_keys(
                keys=_RESTART_CLEAR_KEYS,
                prefixes=_RESTART_CLEAR_PREFIXES)
            st.session_state['uploader_version'] = st.session_state.get('uploader_version', 0) + 1
            st.rerun()


def app():
    """全体の表示制御（5状態）:

    A) 結果表示中（`excel_result`あり）: オプション・領域一覧・
       領域の確認は表示するが操作不可（disabled、振り返り専用）。
       （領域検出パラメータは config.py 側の設定のため、この状態遷移の対象外）
       「領域を検出」「領域選択を完了」「ラベルを抽出」ボタンはいずれも非表示
       （「新しい抽出を開始」でのみ次のサイクルへ進める）。
    B) 領域検出ON・未検出: 「領域を検出」ボタンのみ表示（常に青）。
       「ラベルを抽出」・領域一覧・領域の確認は非表示（region_analysesが無いため）。
    C1) 領域検出ON・検出済み・未確定（「領域選択を完了」未クリック）:
       「領域を検出」は非表示（役目を終えたため）。「領域一覧」（チェックボックス
       群＋「領域選択を完了」ボタン）のみ表示。「領域の確認」「ラベルを抽出」は
       非表示（一括選択が終わるまで、決定できなかった領域が頻繁に増減して
       表示されるのを避けるため）。
    C2) 領域検出ON・検出済み・確定済み（「領域選択を完了」クリック後）:
       「領域選択を完了」ボタンは非表示（役目を終えたため。「領域一覧」の
       チェックを変更すると `_invalidate_region_selection_confirmation` に
       より確定が取り消され C1 に戻る）。「領域の確認」（チェック済み候補が2件以上残る領域のみ）・
       「ラベルを抽出」（常に青）を表示。
    D) 領域検出OFF: 「ラベルを抽出」のみ表示（常に青、既存通り）。
    """
    _render_intro()

    uploaded_files = _render_upload_section()
    if not uploaded_files:
        st.info("DXFファイルをアップロードしてください")
        return
    st.success(f"{len(uploaded_files)}個のファイルが選択されました")

    locked = bool(st.session_state.get('excel_result'))

    opts = _render_options_section(locked)

    region_cfg = _build_region_cfg()

    master_index, master_index_status = _resolve_master_index()

    detected = 'region_analyses' in st.session_state
    region_selection_confirmed = bool(st.session_state.get('region_selection_confirmed'))

    if not locked and opts['enable_region_detection'] and not detected:
        _render_region_detection_trigger(uploaded_files, region_cfg)

    _render_global_region_names_section(locked)

    _render_region_confirmation_section(opts['enable_region_detection'], locked, uploaded_files)

    if not locked and (
        not opts['enable_region_detection']
        or (detected and region_selection_confirmed)
    ):
        _handle_extract_button(uploaded_files, region_cfg, opts, master_index=master_index)

    _render_type_judgment_section(master_index=master_index)
    _render_results_section(master_index_status=master_index_status, uploaded_files=uploaded_files)


if __name__ == "__main__":
    app()
