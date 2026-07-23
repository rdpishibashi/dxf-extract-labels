import streamlit as st
import os
import sys
import contextlib
import traceback
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model')
sys.path.insert(0, model_path)

from model.extract_labels import extract_labels, process_multiple_dxf_files
from model.region_detector import (
    analyze_dxf_regions, build_region_results, DEFAULT_REGION_CONFIG, regions_overlap,
)
from model.excel_output import (
    create_excel_output, create_ref_designator_excel_output, create_region_excel_output,
)
from model.common_utils import save_uploadedfile
from model import ref_designator
from model import decision_log
from model import terminal_detector


def handle_error(e, show_traceback=True):
    """エラーを適切に処理して表示する"""
    st.error(f"エラーが発生しました: {str(e)}")
    if show_traceback:
        st.error(traceback.format_exc())


def _read_github_secrets():
    """st.secrets['github'] を安全に読む（secrets.toml が無い環境では例外を握りつぶす）。"""
    try:
        return st.secrets.get('github')
    except Exception:
        return None

APP_VERSION = '1.9.10'

# 領域境界線の色(ACI)選択肢: AutoCAD標準の基本9色（実データで境界線に使われる
# 色はほぼこの範囲に収まる）。保存済み設定値がこの範囲外（カスタム値）の場合は
# 表示時に動的に追加し、値を失わないようにする（app()内 frm_region_color 参照）。
_ACI_COLOR_CHOICES = [
    (1, "1: 赤"),
    (2, "2: 黄"),
    (3, "3: 緑"),
    (4, "4: シアン"),
    (5, "5: 青"),
    (6, "6: マゼンタ"),
    (7, "7: 白/黒"),
    (8, "8: グレー"),
    (9, "9: 明るいグレー"),
]

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
#   - REGION_DETECT: 「領域を検出」再実行時。ref_pending・未確定ラベル選択
#     状態はこの操作では変化しないため対象外。
#   - EXTRACT: 「ラベルを抽出」実行時。入力状態（region_analyses・uploaded
#     files 等）は保持したまま、前回の抽出結果・未確定ラベル選択状態のみ
#     クリアする。
#   - RESTART: 「新しい抽出を開始」時。入力状態を含む全リセット
#     （オプション設定ウィジェットの値は対象外＝保持される、ユーザー指定）。
# ============================================================
_REGION_DETECT_CLEAR_KEYS = ['excel_result', 'is_region_mode', 'region_results_summary']
_REGION_DETECT_CLEAR_PREFIXES = ('rc_',)

_EXTRACT_CLEAR_KEYS = [
    'excel_result', 'is_region_mode', 'region_results_summary',
    'ref_pending', 'ref_pending_mode', 'ref_results_summary',
    'decision_log_result', 'unclassified_checked', 'unclassified_ver',
    'terminal_results',
]
_EXTRACT_CLEAR_PREFIXES = ('unclassified_editor_',)

_RESTART_CLEAR_KEYS = [
    'excel_result', 'output_filename', 'processing_settings',
    'results', 'is_region_mode', 'region_analyses',
    'region_results_summary', 'ref_pending', 'ref_pending_mode',
    'ref_results_summary', 'download_done', 'decision_log_result',
    'terminal_results',
]
_RESTART_CLEAR_PREFIXES = ('rc_', 'unclassified_editor_')


@contextlib.contextmanager
def _temp_dxf_file(uploaded_file):
    """アップロードされたファイルを一時ファイルに保存し、処理後に削除する。"""
    tmp = save_uploadedfile(uploaded_file)
    try:
        yield tmp
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _set_processing_settings(also_extract_non_circuit, extract_drawing_numbers_option,
                              extract_title_option):
    """「ラベルを抽出」実行時のオプション設定を session_state に記録する
    （結果表示セクションでの再現・ダウンロード時のファイル名等に使う）。"""
    st.session_state.processing_settings = {
        'also_extract_non_circuit': also_extract_non_circuit,
        'extract_drawing_numbers': extract_drawing_numbers_option,
        'extract_title': extract_title_option,
    }


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


def _compute_terminal_results(uploaded_files):
    """「端子一覧を抽出」オプション用に、アップロード済み全ファイルの
    端子台(TB)矩形解析を実行する（`terminal_detector.analyze_dxf_terminals`）。
    対象外ファイル（タイトルが「UNIT内結線図」でない等）は
    `is_target=False` のまま結果に含まれる（Excel出力側で除外される）。"""
    results = {}
    for uf in uploaded_files:
        with _temp_dxf_file(uf) as tmp:
            results[uf.name] = terminal_detector.analyze_dxf_terminals(
                tmp, original_filename=uf.name)
    return results


def _gather_name_selections(fname, analysis):
    """領域確認UIのチェックボックス状態から、この図面の名称選択を集める。"""
    name_selections = {}
    for reg in analysis.get('regions', []):
        chosen = []
        for i, (_d, t) in enumerate(reg['name_candidates']):
            if st.session_state.get(f"rc_{fname}_{reg['id']}_{i}"):
                chosen.append(t)
        if chosen:
            name_selections[(fname, reg['id'])] = chosen
    return name_selections


def _on_change_radio(fname, reg_id, clicked_idx, n_cands, clicked_text):
    """領域名チェックボックスをラジオボタン的に動作させるコールバック。

    チェックが ON になったとき、(1) 同一領域の他チェックボックスを OFF にし、
    (2) 同じ名称候補を持つ他領域（候補が2件以上＝選択肢ありの領域のみ）にも
    同じ名称を選択状態として伝播する。チェックボックスは初回生成時にしか
    デフォルト値を設定できない（st.session_state 既存キーは上書きされない）ため、
    生成後にユーザーが選択を変更したときの伝播はこのコールバックで明示的に行う。

    ただし、同一ファイル内で互いに重なり（完全な内包も部分的な重複も含む）が
    ある領域（例: `EE6313-546-01E.dxf` の `B CHAMBER`〈外側〉と
    `BAKE HEATER UNIT RX`〈内側、完全内包〉）は、まったく別の物理位置を指す
    別個の領域であり、同期してはならない（v1.5.11、ユーザー報告: デフォルトでない
    候補を手動選択すると、重なりのある別領域も同じ名称に同期されてしまう。
    当初は内包のみを対象としていたが、部分的な重複も対象にすべきとの指摘により
    `regions_overlap()` に一般化）。同期は元々 MPD RACK2 のような、空間的に分離
    した（重ならない）複数ピースが同じ名称を共有するケースを想定したもの。
    """
    ck = f"rc_{fname}_{reg_id}_{clicked_idx}"
    if not st.session_state.get(ck, False):
        return
    for j in range(n_cands):
        if j != clicked_idx:
            st.session_state[f"rc_{fname}_{reg_id}_{j}"] = False
    st.session_state[f"rc_{fname}_{reg_id}_none"] = False  # 実候補を選んだら「領域選択しない」を解除

    analyses = st.session_state.get('region_analyses', {})
    clicked_region = next(
        (r for r in analyses.get(fname, {}).get('regions', []) if r['id'] == reg_id), None)
    clicked_polygon = clicked_region['polygon'] if clicked_region else None

    for fn2, an2 in analyses.items():
        for r2 in an2.get('regions', []):
            if fn2 == fname and r2['id'] == reg_id:
                continue
            cands2 = r2.get('name_candidates', [])
            if len(cands2) <= 1:
                continue  # 選択肢なし（自動確定）の領域は対象外
            if not any(t2 == clicked_text for _d2, t2 in cands2):
                continue  # この名称を候補に持たない領域には影響しない
            if (fn2 == fname and clicked_polygon
                    and regions_overlap(clicked_polygon, r2['polygon'])):
                continue  # 重なり（内包/部分重複）のある領域同士は同期しない
            for j2, (_d2, t2) in enumerate(cands2):
                st.session_state[f"rc_{fn2}_{r2['id']}_{j2}"] = (t2 == clicked_text)


def _on_change_none(fname, reg_id, n_cands):
    """「領域選択しない」チェックボックスをラジオボタン的に動作させるコールバック。

    ON になったとき、同一領域の名称候補チェックボックスを全て OFF にする。
    「選択しない」は実在する名称候補ではないため、他領域への同期
    （`_on_change_radio` の (2)）は行わない——この領域だけの個別判断であり、
    同名候補を持つ他領域まで一律「選択しない」にする理由がないため。
    """
    ck = f"rc_{fname}_{reg_id}_none"
    if not st.session_state.get(ck, False):
        return
    for j in range(n_cands):
        st.session_state[f"rc_{fname}_{reg_id}_{j}"] = False


def _compute_default_candidate_index(fname, reg, cands, analyses):
    """他の図面/領域で選択済みの名称があれば、それに一致する候補のインデックスを
    デフォルトとして返す（無ければ0）。

    候補が1件のみ（選択肢なし・自動確定）の領域は同期対象から除外する
    （ユーザーが能動的に選んだわけではない確定のため、隣接領域の候補にも同じ
    ラベルが上がる場合に誤って引き継ぐのを防ぐ）。

    この領域自身の最有力候補が Tier1/2（下端/上端最近傍。回転図面では右端/左端）
    の確信度の高い候補である場合は、他領域からの同期で上書きしない（v1.5.9）。
    同期は元々「この領域自身に強い候補が無く（Tier3）、他の領域で選ばれた同名
    候補を引き継ぐべき」ケース（例: MPD RACK2 のような複数ピース合算）のために
    設けたもので、隣接・入れ子の領域がそれぞれ独自の Tier1/2 候補を持つ場合に
    互いの選択を上書きしてしまう不具合があった（ユーザー報告: EE6313-546-01E.dxf
    の図面1/領域1,2 が同じ選択に同期されるが、本来は領域1=B CHAMBER、
    領域2=BAKE HEATER UNIT RX で別々が正しい）。

    同一ファイル内で互いに重なり（完全な内包も部分的な重複も含む）がある領域
    同士は、別個の領域として扱い同期しない（v1.5.11、ユーザー報告: デフォルト
    でない候補を手動選択すると、重なりのある別領域も同じ名称に同期されてしまう。
    当初は内包のみを対象としていたが、部分的な重複も対象にすべきとの指摘により
    `regions_overlap()` に一般化）。
    """
    if reg.get('default_name_tier') in (1, 2):
        return 0
    selected_elsewhere = set()
    for fn2, an2 in analyses.items():
        for r2 in an2.get('regions', []):
            if fn2 == fname and r2['id'] == reg['id']:
                continue
            cands2 = r2.get('name_candidates', [])
            if len(cands2) <= 1:
                continue  # 選択肢なし（自動確定）の領域はスキップ
            if fn2 == fname and regions_overlap(reg['polygon'], r2['polygon']):
                continue  # 重なり（内包/部分重複）のある領域同士は同期しない
            for j, (_d2, t2) in enumerate(cands2):
                if st.session_state.get(f"rc_{fn2}_{r2['id']}_{j}", False):
                    selected_elsewhere.add(t2)
    for i, (_d, t) in enumerate(cands):
        if t in selected_elsewhere:
            return i
    return 0


def _render_corners_popover(corners):
    """領域の頂点座標一覧を📐ポップオーバーで表示する。"""
    with st.popover("📐"):
        st.markdown(f"**頂点の座標**（左下から / {len(corners)}点）")
        st.code(
            '\n'.join(
                f"{i + 1}: ({x:.2f}, {y:.2f})"
                for i, (x, y) in enumerate(corners)
            ) or '(なし)'
        )


def _render_dangling_edges_section(region_dangling):
    """この領域の行き止まり枝（境界探索から除外された未閉路の線分群）を
    ⚠️エクスパンダーで表示する。`region_dangling` が空なら何もしない。"""
    if not region_dangling:
        return
    with st.expander(f"⚠️ この領域の行き止まり枝（{len(region_dangling)} 件）"):
        st.caption(
            "この領域の境界探索から除外された、どこにも閉じていない"
            "線分（境界線と同じ線種 lineweight=25/color=2）です。"
            "手描きの作画ミスの可能性があるため、該当する handle を"
            "確認してください。"
        )
        lines = []
        for br in region_dangling:
            att = br.get('attachment')
            att_str = f"({att[0]:.2f}, {att[1]:.2f})" if att else "(不明)"
            lines.append(f"取り付け点 {att_str}:")
            for ent in br['entities']:
                h = ent['handle'] or '(handle不明)'
                (sx, sy), (ex, ey) = ent['start'], ent['end']
                lines.append(
                    f"  handle {h}: "
                    f"({sx:.2f}, {sy:.2f}) - ({ex:.2f}, {ey:.2f})"
                )
        st.code('\n'.join(lines))


def app():
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
            "既定では、図面枠内の機器符号（Reference Designator）パターンに一致するラベルのみを"
            "候補として抽出します。",
            "",
            "**主なオプション：**",
            "- 機器符号（候補）以外も抽出：OFF（既定）は図面枠内・機器符号パターンに一致する"
            "ラベルのみを候補として抽出し、未確定ラベルを人手で選択します。ONにすると"
            "図面枠・パターンの制限なしに全ラベルを抽出します（この場合、未確定ラベルの"
            "選択は行いません）",
            "- 図面番号・タイトル・サブタイトルを抽出：図番・タイトル・サブタイトルをSummary"
            "シートに記録します",
            "- 領域を検出：図面枠内の矩形領域を検出し、領域ごとにラベルを分類します",
            "- 端子一覧を抽出：「UNIT内結線図」の図面を対象に、端子台の端子番号を"
            "「TB List」シートに出力します",
            "",
            "**使用手順（既定モード）：**",
            "1. DXFファイルをアップロードしてください（複数可）",
            "2. 必要に応じてオプションを設定します",
            "3. 「領域を検出」をONにした場合は、「領域を検出」ボタンで矩形領域を検出し、"
            "各領域の名称を確認・選択します",
            "4. 「ラベルを抽出」ボタンで処理を実行します",
            "5. 「未確定ラベル」でReference Designatorとして採用するラベルにチェックを入れ、"
            "「選択完了」を押します（「機器符号（候補）以外も抽出」ON時はこの手順は不要です）",
            "",
            "**Excelファイルの内容：**",
            "- Summaryシート：全ファイルの抽出結果概要",
            "- Totalシート（通常モード）／領域一覧・領域別ラベル一覧シート（領域検出時）："
            "全ファイル横断のラベル集計",
            "- 各ファイルシート：個別ファイルの抽出ラベル一覧（領域検出時は「領域」列付き）",
            "- TB Listシート（端子一覧を抽出時）：端子台ごとの端子番号一覧",
        ]))

    # ============================================================
    # ファイルアップロード
    # ============================================================
    st.subheader("DXFファイルのアップロード")
    if 'uploader_version' not in st.session_state:
        st.session_state['uploader_version'] = 0
    uploaded_files = st.file_uploader(
        "DXFファイルを選択してください（複数可）",
        type="dxf",
        accept_multiple_files=True,
        key=f"dxf_uploader_{st.session_state['uploader_version']}",
    )

    if not uploaded_files:
        st.info("DXFファイルをアップロードしてください")
        return

    st.success(f"{len(uploaded_files)}個のファイルが選択されました")

    # ============================================================
    # オプション
    # ============================================================
    st.subheader("オプション")
    col_left, col_right = st.columns(2)

    with col_left:
        # a) 機器符号（候補）以外も抽出（デフォルト OFF = 機器符号（候補）のみ）
        also_extract_non_circuit = st.checkbox(
            "機器符号（候補）以外も抽出",
            value=False,
            help="チェックなし（既定）：図面枠内・図面情報欄外のラベルのうち、機器符号\n"
                 "（Reference Designator）パターンに一致し除外パターンに該当しないものが対象。\n"
                 "確実に機器符号と判定できる形（確定パターン）に一致したものは自動採用され、\n"
                 "それ以外は「未確定ラベル」として一覧表示し、チェックして採用したものだけが\n"
                 "最終的に機器符号として出力されます。\n\n"
                 "チェックあり：図面枠外・図面情報欄内も含め、すべてのラベルを"
                 "そのまま抽出します（機器符号（候補）の判定・未確定ラベルの選択は行いません）。\n\n"
                 "【機器符号パターン例】\n"
                 "• 英文字のみ: CNCNT, FB\n"
                 "• 英文字+数字+ハイフン任意: R10, CN3, AAC1B4-07\n"
                 "• 英字-英字+数字: CN-IF2-1\n"
                 "• 括弧付き（括弧より前で判定）: FB(), MSS(MOTOR), R10(2.2K)"
        )

        # b) 図面番号・タイトル・サブタイトルを抽出（統合）
        extract_all_option = st.checkbox(
            "図面番号・タイトル・サブタイトルを抽出",
            value=False,
            help="図面番号（例：EE6868-500-01C）、タイトル、サブタイトルをサマリーシートに抽出します。"
        )
        extract_drawing_numbers_option = extract_all_option
        extract_title_option = extract_all_option

        # c) 領域を検出（矩形領域抽出機能全体のON/OFF）
        enable_region_detection = st.checkbox(
            "領域を検出",
            value=False,
            key='enable_region_detection',
            help="ONにすると「領域検出の詳細設定」「領域を検出」ボタン「領域の確認」"
                 "セクションが表示され、矩形領域付きでラベルを抽出できます。"
                 "OFFにすると通常モード（機器符号のみ抽出等）になります。"
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
                keys=['region_analyses'],
                prefixes=_REGION_DETECT_CLEAR_PREFIXES)

        # d) 端子一覧を抽出
        extract_terminal_option = st.checkbox(
            "端子一覧を抽出",
            value=False,
            help="タイトルが「UNIT内結線図」の図面を対象に、TBで始まり英大文字・数字が"
                 "続く端子台ラベルに対応する矩形（LINE+CIRCLEで構成）を検出し、矩形内の"
                 "端子番号を「TB List」シート（Totalシート直後）に出力します。"
                 "「端子台」でユニークをとり、同じ番号が複数ある場合は件数をカッコで"
                 "表示します（例: 7(2)）。\n\n"
                 "サブタイトルが「TB COMPONENT」の図面（端子台の全端子番号一覧ページ）"
                 "は対象外です。"
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
            index=0
        )
        sort_value = sort_option[1]

        output_filename = st.text_input(
            "出力Excelファイル名",
            value="extracted_labels.xlsx",
            help="出力するExcelファイルの名前を指定します"
        )
        if not output_filename.endswith('.xlsx'):
            output_filename += '.xlsx'

    # ============================================================
    # 領域検出の詳細設定（フォーム形式 — 「設定完了」で確定）
    # ============================================================
    _cfg_defaults = dict(DEFAULT_REGION_CONFIG)
    _cfg_defaults['name_max_dist'] = 10.0  # デフォルトを10に変更
    saved_cfg = st.session_state.get('saved_region_cfg', _cfg_defaults)
    region_cfg = dict(saved_cfg)  # 「設定完了」が押されるまではこの値を使用

    submitted = False
    if enable_region_detection:
        with st.expander("領域検出の詳細設定", expanded=False):
            st.caption("値を変更した後、「設定完了」ボタンを押すと確定されます（ボタンを押さずに変更した値は反映されません）。")
            with st.form("region_cfg_form"):
                rc1, rc2 = st.columns(2)
                with rc1:
                    frm_frame_lw = st.number_input(
                        "図面枠の太さ",
                        value=int(saved_cfg['frame_lineweight']),
                        step=5,
                        help="図面全体を囲む枠の線の太さ（lineweight）",
                        key='frm_frame_lineweight')
                    frm_region_lw = st.number_input(
                        "領域境界線の太さ",
                        value=int(saved_cfg['region_lineweight']),
                        step=5,
                        help="矩形領域の境界線の太さ（lineweight）",
                        key='frm_region_lineweight')
                    frm_cp_margin = st.number_input(
                        "接続点判定マージン（座標）",
                        value=float(saved_cfg['connection_point_margin']),
                        min_value=0.0, step=0.01, format="%.2f",
                        help="接続点（円）が境界線からこの座標距離以内なら「境界上」とみなします。"
                             "縦ギャップ上に接続点がある場合の橋渡し除外にも使用します。",
                        key='frm_connection_point_margin')
                    _current_region_color = int(saved_cfg['region_color'])
                    _region_color_choices = list(_ACI_COLOR_CHOICES)
                    if _current_region_color not in dict(_region_color_choices):
                        _region_color_choices = _region_color_choices + [
                            (_current_region_color, f"{_current_region_color}: (カスタム)")]
                    frm_region_color = st.selectbox(
                        "領域境界線の色(ACI)",
                        options=[v for v, _ in _region_color_choices],
                        index=[v for v, _ in _region_color_choices].index(_current_region_color),
                        format_func=lambda v: dict(_region_color_choices)[v],
                        help="矩形領域の境界線の色（既定: 2 = 黄）",
                        key='frm_region_color')
                    frm_filter_prefixes = st.text_input(
                        "抽出したい領域名の先頭文字列（カンマ区切り・複数可）",
                        value=','.join(saved_cfg.get('name_filter_prefixes', ())),
                        help="指定すると、名称候補がこの文字列のいずれかで始まる領域は、"
                             "面積比の閾値を満たさなくても抽出対象になります"
                             "（前方一致。全角/半角は区別しません）。空欄なら従来通り"
                             "面積比のみで判定します。",
                        key='frm_name_filter_prefixes')
                    frm_terms = st.text_input(
                        "領域名称候補から除外する単語（カンマ区切り）",
                        value=','.join(saved_cfg['name_exclude_terms']),
                        help="これらの語を含むラベルを名称候補から除外します",
                        key='frm_name_exclude_terms')
                with rc2:
                    frm_area_pct = st.number_input(
                        "最小面積（単独領域・図面枠面積比 %）",
                        value=min(99, max(1, round(saved_cfg['area_ratio'] * 100))),
                        min_value=1, max_value=99, step=1,
                        help="1つの閉領域が単独でこの面積比（四捨五入した整数%）以上のとき抽出対象とします",
                        key='frm_area_ratio_pct')
                    frm_group_pct = st.number_input(
                        "最小面積（同名複数領域・図面枠面積比 %）",
                        value=min(50, max(1, round(saved_cfg['group_area_ratio'] * 100))),
                        min_value=1, max_value=50, step=1,
                        help="同じ名称の複数ピースを合算したとき、この面積比（四捨五入した整数%）以上なら抽出対象とします。"
                             "第1図面で成立した名称は他図面でも面積不問で抽出します。",
                        key='frm_group_area_ratio_pct')
                    frm_name_max = st.number_input(
                        "領域名称ラベルの境界線からの最大距離（座標）",
                        value=min(20, max(1, int(saved_cfg.get('name_max_dist', 10.0)))),
                        min_value=1, max_value=20, step=1,
                        help="下端境界線からこの座標距離以内のラベルを名称候補とします",
                        key='frm_name_max_dist')
                    frm_name_min = st.number_input(
                        "領域名称ラベルの境界線からの最小距離（座標）",
                        value=min(10, max(1, int(saved_cfg['name_min_dist']))),
                        min_value=1, max_value=10, step=1,
                        help="この距離未満（境界線分上）のラベルは名称候補から除外します",
                        key='frm_name_min_dist')
                    frm_min_letters = st.number_input(
                        "領域名称候補に必要な文字数",
                        value=int(saved_cfg['name_min_letters']),
                        min_value=1, step=1,
                        help="英字がこの文字数以上のラベルのみ名称候補とします",
                        key='frm_name_min_letters')

                submitted = st.form_submit_button("設定完了", type="primary")

        # フォーム送信時に設定を確定・保存
        if submitted:
            region_cfg.update({
                'frame_lineweight': frm_frame_lw,
                'region_lineweight': frm_region_lw,
                'region_color': frm_region_color,
                'connection_point_margin': frm_cp_margin,
                'area_ratio': frm_area_pct / 100.0,
                'group_area_ratio': frm_group_pct / 100.0,
                'name_max_dist': float(frm_name_max),
                'name_min_dist': float(frm_name_min),
                'name_min_letters': frm_min_letters,
                'name_exclude_terms': tuple(
                    s.strip() for s in frm_terms.split(',') if s.strip()),
                'name_filter_prefixes': tuple(
                    s.strip() for s in frm_filter_prefixes.split(',') if s.strip()),
            })
            st.session_state['saved_region_cfg'] = dict(region_cfg)
            st.session_state['region_cfg_is_saved'] = True
            st.toast("設定を保存しました", icon="✅")

    if enable_region_detection and st.session_state.get('region_cfg_is_saved'):
        st.caption("✅ 詳細設定：保存済み")

    # ============================================================
    # 領域を検出
    # ============================================================
    if enable_region_detection:
        detect_done = 'region_analyses' in st.session_state
        # 「ラベルを抽出」が開始済み（未確定ラベルの選択待ち含む）なら、領域検出は
        # もはやこの回の抽出には反映されないため「領域を検出」を白にする。
        extract_done = bool(st.session_state.get('excel_result')) or bool(st.session_state.get('ref_pending'))
        detect_btn_type = "secondary" if (detect_done or extract_done) else "primary"
        if st.button("領域を検出", key="detect_regions_btn", type=detect_btn_type):
            try:
                with st.spinner('図面枠と矩形領域を検出中...'):
                    analyses = {}
                    for uf in uploaded_files:
                        with _temp_dxf_file(uf) as tmp:
                            analysis = analyze_dxf_regions(tmp, region_cfg)
                            if extract_all_option:
                                _, dn_info = extract_labels(
                                    tmp,
                                    extract_drawing_numbers_option=True,
                                    extract_title_option=True,
                                    original_filename=uf.name,
                                )
                                analysis['main_drawing_number'] = dn_info.get('main_drawing_number')
                                analysis['title'] = dn_info.get('title')
                                analysis['subtitle'] = dn_info.get('subtitle')
                            analyses[uf.name] = analysis
                    st.session_state['region_analyses'] = analyses
                    _clear_session_keys(
                        keys=_REGION_DETECT_CLEAR_KEYS,
                        prefixes=_REGION_DETECT_CLEAR_PREFIXES)
                st.rerun()
            except Exception as e:
                handle_error(e)

    # ============================================================
    # 領域の確認
    # ============================================================
    if enable_region_detection and 'region_analyses' in st.session_state:
        analyses = st.session_state['region_analyses']
        st.subheader("領域の確認")

        for file_idx, (fname, analysis) in enumerate(sorted(analyses.items())):
            if file_idx > 0:
                st.divider()
            st.markdown(f"### {fname}")
            dn_parts = []
            if analysis.get('main_drawing_number'):
                dn_parts.append(f"図番：{analysis['main_drawing_number']}")
            if analysis.get('title'):
                dn_parts.append(f"タイトル：{analysis['title']}")
            if analysis.get('subtitle'):
                dn_parts.append(f"サブタイトル：{analysis['subtitle']}")
            if dn_parts:
                st.caption("　/　".join(dn_parts))
            err = analysis.get('error')
            n_frames = len(analysis.get('frames', []))
            regions = analysis.get('regions', [])
            st.caption(f"図面枠 {n_frames} 個 / 検出領域 {len(regions)} 個")
            if err:
                st.warning(err)
                continue

            if not regions:
                st.info("面積条件を満たす領域が検出されませんでした。詳細設定の調整をお試しください。")
                continue

            for reg_idx, reg in enumerate(regions):
                if reg_idx > 0:
                    st.divider()

                corners = reg.get('corners', [])
                # 「図面#/領域#（面積 xx%）」 と 📐 ボタンを同じ行に
                col_header, col_btn = st.columns([8, 1])
                with col_header:
                    st.markdown(
                        f"　**図面{reg['frame'] + 1} / 領域{reg['id'] + 1}**"
                        f"　面積 {reg['area_pct']:.0f}%"
                    )
                with col_btn:
                    _render_corners_popover(corners)

                _render_dangling_edges_section(reg.get('dangling_edges', []))

                cands = reg['name_candidates']
                if not cands:
                    st.caption("　　（名称候補なし）")
                    continue

                default_idx = _compute_default_candidate_index(fname, reg, cands, analyses)

                for i, (d, t) in enumerate(cands):
                    ck = f"rc_{fname}_{reg['id']}_{i}"
                    if ck not in st.session_state:
                        st.session_state[ck] = (i == default_idx)
                    st.checkbox(
                        f"　{t}　（境界線からの距離 {d:.0f}）",
                        key=ck,
                        on_change=_on_change_radio,
                        args=(fname, reg['id'], i, len(cands), t)
                    )

                # 領域選択しない（この領域にはどの名称候補も割り当てない）。
                # 既定は未チェック（従来通りいずれかの候補が既定選択される）で、
                # ユーザーが明示的に選んだときだけこの領域を無名扱いにする。
                ck_none = f"rc_{fname}_{reg['id']}_none"
                if ck_none not in st.session_state:
                    st.session_state[ck_none] = False
                st.checkbox(
                    "　（領域選択しない）",
                    key=ck_none,
                    on_change=_on_change_none,
                    args=(fname, reg['id'], len(cands))
                )

    # ============================================================
    # ラベルを抽出
    # ============================================================
    has_results = bool(st.session_state.get('excel_result')) or bool(st.session_state.get('ref_pending'))
    extract_btn_type = "secondary" if has_results else "primary"
    if st.button("ラベルを抽出", type=extract_btn_type):
        try:
            frame_lineweight = int(region_cfg['frame_lineweight'])
            with st.spinner(f'{len(uploaded_files)}個のDXFファイルを処理中...'):
                # 前回の結果・未確定ラベル選択状態をクリアする
                _clear_session_keys(
                    keys=_EXTRACT_CLEAR_KEYS,
                    prefixes=_EXTRACT_CLEAR_PREFIXES)

                # 「端子一覧を抽出」: 「選択完了」待ちのモードでも使えるよう、
                # ここで一度だけ解析して session_state に保持する。
                if extract_terminal_option:
                    st.session_state['terminal_results'] = _compute_terminal_results(uploaded_files)
                terminal_results = st.session_state.get('terminal_results')

                if 'region_analyses' in st.session_state:
                    # 領域付きモード
                    analyses = st.session_state['region_analyses']

                    if also_extract_non_circuit:
                        # 図面枠制限・機器符号フィルタなしで全ラベルを対象にする
                        name_selections = {}
                        for fname, analysis in analyses.items():
                            name_selections.update(_gather_name_selections(fname, analysis))
                        region_results = build_region_results(
                            analyses, name_selections, sort_value, filter_circuit_only=False)

                        st.session_state.excel_result = create_region_excel_output(
                            region_results, terminal_results=terminal_results)
                        st.session_state.output_filename = output_filename
                        st.session_state['is_region_mode'] = True
                        st.session_state['region_results_summary'] = {
                            f: {k: v for k, v in d.items() if k not in ('named', 'rows')}
                            for f, d in region_results.items()
                        }
                        _set_processing_settings(True, extract_drawing_numbers_option, extract_title_option)
                    else:
                        # 機器符号（候補）パイプライン: 未確定ラベルの選択待ちにする
                        ref_pending = {}
                        for fname, analysis in analyses.items():
                            uf = next(f for f in uploaded_files if f.name == fname)
                            with _temp_dxf_file(uf) as tmp:
                                data = ref_designator.extract_ref_designator_data(
                                    tmp, frame_lineweight=frame_lineweight, original_filename=fname)
                            data['main_drawing_number'] = analysis.get('main_drawing_number')
                            data['title'] = analysis.get('title')
                            data['subtitle'] = analysis.get('subtitle')
                            ref_pending[fname] = data
                        st.session_state['ref_pending'] = ref_pending
                        st.session_state['ref_pending_mode'] = 'region'
                        st.session_state.output_filename = output_filename
                        _set_processing_settings(False, extract_drawing_numbers_option, extract_title_option)

                else:
                    # 通常モード
                    if also_extract_non_circuit:
                        original_filenames = [f.name for f in uploaded_files]
                        with contextlib.ExitStack() as stack:
                            temp_files = [stack.enter_context(_temp_dxf_file(f))
                                         for f in uploaded_files]
                            results_temp = process_multiple_dxf_files(
                                temp_files,
                                sort_order=sort_value,
                                extract_drawing_numbers_option=extract_drawing_numbers_option,
                                extract_title_option=extract_title_option,
                                original_filenames=original_filenames,
                            )

                            results = {}
                            for temp_file, original_name in zip(temp_files, original_filenames):
                                if temp_file in results_temp:
                                    labels, info = results_temp[temp_file]
                                    info['filename'] = original_name
                                    results[original_name] = (labels, info)

                        st.session_state.excel_result = create_excel_output(
                            results, sort_value, terminal_results=terminal_results)
                        st.session_state.output_filename = output_filename
                        st.session_state['is_region_mode'] = False
                        st.session_state.results = results
                        _set_processing_settings(True, extract_drawing_numbers_option, extract_title_option)
                    else:
                        # 機器符号（候補）パイプライン: 未確定ラベルの選択待ちにする
                        ref_pending = {}
                        for uploaded_file in uploaded_files:
                            with _temp_dxf_file(uploaded_file) as tmp:
                                data = ref_designator.extract_ref_designator_data(
                                    tmp, frame_lineweight=frame_lineweight,
                                    original_filename=uploaded_file.name)
                                if extract_all_option:
                                    _, dn_info = extract_labels(
                                        tmp,
                                        extract_drawing_numbers_option=True,
                                        extract_title_option=True,
                                        original_filename=uploaded_file.name,
                                    )
                                    data['main_drawing_number'] = dn_info.get('main_drawing_number')
                                    data['source_drawing_number'] = dn_info.get('source_drawing_number')
                                    data['title'] = dn_info.get('title')
                                    data['subtitle'] = dn_info.get('subtitle')
                            ref_pending[uploaded_file.name] = data
                        st.session_state['ref_pending'] = ref_pending
                        st.session_state['ref_pending_mode'] = 'normal'
                        st.session_state.output_filename = output_filename
                        _set_processing_settings(False, extract_drawing_numbers_option, extract_title_option)

                st.session_state['download_done'] = False

            st.rerun()
        except Exception as e:
            handle_error(e)

    # ============================================================
    # 未確定ラベル（機器符号（候補）パイプラインのみ、選択完了までの間表示）
    # ============================================================
    # 列幅（文字数の目安: 採用=4字, ラベル=8字, 個数=3字の75%幅）。テーブル全体も
    # 固定幅にし、st.container(horizontal=True) の折り返しでブラウザー幅に応じて
    # テーブルの横並び数（狭ければ2列、広ければ3列、さらに広ければ4列…）が
    # 自動調整されるようにする。テーブル数は固定せず、テーブルあたりの行数を
    # 固定してデータ量に応じた個数のテーブルを生成することで、折り返しが
    # ブラウザー幅いっぱいまで機能する。
    UNCLASSIFIED_COL_WIDTH_APPROVE = 68   # 元90pxの75%
    UNCLASSIFIED_COL_WIDTH_LABEL = 120    # 元160pxの75%
    UNCLASSIFIED_COL_WIDTH_COUNT = 53     # 元70pxの75%
    UNCLASSIFIED_TABLE_WIDTH = (
        UNCLASSIFIED_COL_WIDTH_APPROVE + UNCLASSIFIED_COL_WIDTH_LABEL
        + UNCLASSIFIED_COL_WIDTH_COUNT + 20
    )
    UNCLASSIFIED_ROWS_PER_TABLE = 10

    if st.session_state.get('ref_pending'):
        ref_pending = st.session_state['ref_pending']
        st.subheader("未確定ラベル")
        st.caption(
            "機器符号（候補）パターンに一致し、除外パターンに該当しなかったラベルのうち、"
            "確定パターン（レビュー不要で自動採用される形）に一致しなかったものです。"
            "Reference Designator として採用するものにチェックを入れ、"
            "「選択完了」を押してください（チェックしたものだけが出力されます）。"
            "末尾の数字（1〜2桁）を除いた部分が同じラベル（例: CN1・CN2・CN10）と"
            "同一ラベルは、すべてのファイルにわたり連動して採用/解除されます。"
        )

        # チェック状態の正本。data_editor のウィジェット状態は直接書き換えず、
        # この dict を更新して unclassified_ver（キーのバージョン）を上げることで
        # 次の run で新しいウィジェットとして再生成する（生きているインスタンスの
        # 状態書き換えはブラウザ/サーバー間の同期ループを起こすため行わない）。
        checked_state = st.session_state.setdefault('unclassified_checked', {})
        editor_ver = st.session_state.setdefault('unclassified_ver', 0)

        edited_frames = {}
        for file_idx, (fname, data) in enumerate(sorted(ref_pending.items())):
            if file_idx > 0:
                st.divider()
            st.markdown(f"### {fname}")
            if data.get('warning'):
                st.warning(data['warning'])
            review_rows = ref_designator.build_labeled_rows(data['review_labels'])
            st.caption(
                f"図面枠内ラベル数 {data['total_in_frame']}　/　"
                f"確定（自動採用） {len(data['confirmed_labels'])}　/　"
                f"未確定ラベル（要選択） {len(review_rows)} 種"
            )
            if not review_rows:
                st.caption("　　（未確定ラベルなし）")
                continue

            file_checked = checked_state.setdefault(fname, {})
            for row in review_rows:
                file_checked.setdefault(row['ラベル'], False)

            # 固定幅・固定行数のテーブルに分けて表示する。st.container(horizontal=True)
            # によりブラウザー幅に収まる個数だけ横に並び、収まらない分は自動的に
            # 次の行へ折り返す（ブラウザーが広いほど多くのテーブルが横に並ぶ）。
            groups = [
                review_rows[i:i + UNCLASSIFIED_ROWS_PER_TABLE]
                for i in range(0, len(review_rows), UNCLASSIFIED_ROWS_PER_TABLE)
            ]
            dfs = []
            with st.container(horizontal=True):
                for suffix, rows in enumerate(groups):
                    if not rows:
                        continue
                    d = pd.DataFrame(rows)
                    d.insert(0, '採用', [file_checked[r['ラベル']] for r in rows])
                    edited = st.data_editor(
                        d,
                        key=f"unclassified_editor_{editor_ver}_{fname}_{suffix}",
                        column_config={
                            '採用': st.column_config.CheckboxColumn(
                                '採用', default=False, width=UNCLASSIFIED_COL_WIDTH_APPROVE),
                            'ラベル': st.column_config.TextColumn(
                                'ラベル', disabled=True, width=UNCLASSIFIED_COL_WIDTH_LABEL),
                            '個数': st.column_config.NumberColumn(
                                '個数', disabled=True, width=UNCLASSIFIED_COL_WIDTH_COUNT),
                        },
                        hide_index=True,
                        width=UNCLASSIFIED_TABLE_WIDTH,
                    )
                    dfs.append(edited)
            edited_frames[fname] = dfs

        # チェック変更の検出と兄弟ラベルへの連動（正本を更新→キー更新→再描画）。
        # 差分がない run（再描画直後・選択完了クリック時）は何もしないため、
        # 再実行が連鎖することはない。
        deltas = []
        for fname, dfs in edited_frames.items():
            file_checked = checked_state.get(fname, {})
            for edited_df in dfs:
                for lbl, val in zip(edited_df['ラベル'], edited_df['採用']):
                    if bool(val) != file_checked.get(lbl, False):
                        deltas.append((fname, lbl, bool(val)))
        if deltas:
            for _fname, lbl, val in deltas:
                ref_designator.propagate_selection_all_files(checked_state, lbl, val)
            st.session_state['unclassified_ver'] = editor_ver + 1
            st.rerun()

        if st.button("選択完了", type="primary"):
            try:
                with st.spinner("選択内容を反映しています..."):
                    approved_by_file = {}
                    for fname in ref_pending:
                        approved_by_file[fname] = {
                            lbl for lbl, v in checked_state.get(fname, {}).items() if v
                        }

                    # 判断ログ: 未確定ラベルの採用/非採用をエントリ化する
                    # （記録自体は Excel 生成が成功した後に行う）
                    log_entries = []
                    for fname, data in ref_pending.items():
                        log_entries += decision_log.build_entries(
                            file_name=fname,
                            drawing_number=data.get('main_drawing_number'),
                            review_labels=data['review_labels'],
                            approved=approved_by_file[fname],
                            app_version=APP_VERSION,
                            patterns_version=ref_designator.PATTERNS_VERSION,
                        )

                    if st.session_state.get('ref_pending_mode') == 'region':
                        analyses = st.session_state['region_analyses']
                        name_selections_by_file = {
                            fname: _gather_name_selections(fname, analyses[fname])
                            for fname in ref_pending
                        }
                        region_results = ref_designator.build_region_results_from_pending(
                            ref_pending, analyses, name_selections_by_file,
                            approved_by_file, sort_value)
                        st.session_state.excel_result = create_region_excel_output(
                            region_results, terminal_results=st.session_state.get('terminal_results'))
                        st.session_state['is_region_mode'] = True
                        st.session_state['region_results_summary'] = {
                            f: {k: v for k, v in d.items() if k not in ('named', 'rows')}
                            for f, d in region_results.items()
                        }
                    else:
                        ref_final = ref_designator.build_ref_final_from_pending(
                            ref_pending, approved_by_file, sort_value)
                        st.session_state.excel_result = create_ref_designator_excel_output(
                            ref_final, sort_value,
                            terminal_results=st.session_state.get('terminal_results'))
                        st.session_state['is_region_mode'] = False
                        st.session_state['ref_results_summary'] = ref_final

                    # 判断ログの記録（失敗しても抽出本体は止めない。結果は
                    # rerun 後の抽出結果セクションで表示する）
                    log_ok, log_msg = decision_log.record(
                        log_entries, github_conf=_read_github_secrets())
                    st.session_state['decision_log_result'] = {
                        'ok': log_ok,
                        'message': log_msg,
                        'fallback_csv': (
                            None if log_ok
                            else decision_log.entries_to_csv_bytes(log_entries)
                        ),
                    }

                    del st.session_state['ref_pending']
                    st.session_state.pop('ref_pending_mode', None)
                    st.session_state['download_done'] = False
                st.rerun()
            except Exception as e:
                handle_error(e)

    # ============================================================
    # 抽出結果
    # ============================================================
    if st.session_state.get('excel_result'):
        is_region_mode = st.session_state.get('is_region_mode', False)
        settings = st.session_state.get('processing_settings', {})
        candidate_mode = not settings.get('also_extract_non_circuit', False)

        if is_region_mode:
            summ = st.session_state.get('region_results_summary', {})
            st.success(f"{len(summ)}個のDXFファイルからラベル抽出が完了しました（領域付き）")
        elif candidate_mode:
            summ = st.session_state.get('ref_results_summary', {})
            st.success(f"{len(summ)}個のDXFファイルから機器符号（候補）の抽出が完了しました")
        else:
            results = st.session_state.get('results', {})
            st.success(f"{len(results)}個のDXFファイルからラベル抽出が完了しました")

        log_result = st.session_state.get('decision_log_result')
        if log_result:
            if log_result['ok']:
                st.caption(f"📝 {log_result['message']}")
            else:
                st.warning(f"📝 {log_result['message']}")
                if log_result.get('fallback_csv'):
                    st.download_button(
                        label="判断ログをダウンロード（記録失敗分）",
                        data=log_result['fallback_csv'],
                        file_name="decision_log_fallback.csv",
                        mime="text/csv",
                        key="decision_log_fallback_download",
                    )

        with st.expander("📊 抽出結果統計", expanded=False):
            if is_region_mode:
                summ = st.session_state.get('region_results_summary', {})
                if summ:
                    rows = []
                    for f, d in summ.items():
                        rows.append({
                            'ファイル名': f,
                            '総抽出ラベル数': d.get('total_in_frame', 0),
                            '除外ラベル数': d.get('filtered_count', 0),
                            '最終ラベル数': d.get('final_count', 0),
                            '図面枠数': d.get('frames', 0),
                            '検出領域数': d.get('regions_detected', 0),
                            '確定領域数': d.get('regions_named', 0),
                            '領域内ラベル数': d.get('in_region_count', 0),
                        })
                    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            elif candidate_mode:
                summ = st.session_state.get('ref_results_summary', {})
                rows = []
                for f, d in summ.items():
                    row = {
                        'ファイル名': f,
                        '図面枠内ラベル数': d.get('total_in_frame', 0),
                        '機器符号（候補）数': sum(r['個数'] for r in d.get('rows', [])),
                        '未確定ラベル数（未採用）': d.get('unclassified_count', 0),
                    }
                    if settings.get('extract_drawing_numbers'):
                        row['図番'] = d.get('main_drawing_number', '') or ''
                        row['流用元図番'] = d.get('source_drawing_number', '') or ''
                    if settings.get('extract_title'):
                        row['タイトル'] = d.get('title', '') or ''
                        row['サブタイトル'] = d.get('subtitle', '') or ''
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            else:
                results = st.session_state.get('results', {})
                rows = []
                for file_path, (labels, info) in results.items():
                    filename = info.get('filename', os.path.basename(file_path))
                    row = {
                        'ファイル名': filename,
                        '総抽出ラベル数': info.get('total_extracted', 0),
                        '最終ラベル数': info.get('final_count', 0),
                    }
                    if settings.get('extract_drawing_numbers'):
                        row['図番'] = info.get('main_drawing_number', '')
                        row['流用元図番'] = info.get('source_drawing_number', '')
                    if settings.get('extract_title'):
                        row['タイトル'] = info.get('title', '')
                        row['サブタイトル'] = info.get('subtitle', '')
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        download_done = st.session_state.get('download_done', False)

        st.write(f"出力ファイル：**{st.session_state.output_filename}**")
        downloaded = st.download_button(
            label="Excelをダウンロード",
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


if __name__ == "__main__":
    app()
