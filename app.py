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
    collect_unique_region_candidate_names, resolve_globally_decided_name,
    is_region_pending_selection,
)
from model.excel_output import (
    create_excel_output, create_ref_designator_excel_output, create_region_excel_output,
)
from model.common_utils import save_uploadedfile
from model import ref_designator
from model import terminal_detector


def handle_error(e, show_traceback=True):
    """エラーを適切に処理して表示する"""
    st.error(f"エラーが発生しました: {str(e)}")
    if show_traceback:
        st.error(traceback.format_exc())

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
#   - REGION_DETECT: 「領域を検出」再実行時。
#   - EXTRACT: 「ラベルを抽出」実行時。入力状態（region_analyses・uploaded
#     files 等）は保持したまま、前回の抽出結果のみクリアする。
#   - RESTART: 「新しい抽出を開始」時。入力状態を含む全リセット
#     （オプション設定ウィジェットの値は対象外＝保持される、ユーザー指定）。
# ============================================================
_REGION_DETECT_CLEAR_KEYS = [
    'excel_result', 'is_region_mode', 'region_results_summary',
    'global_region_checked_names',
]
_REGION_DETECT_CLEAR_PREFIXES = ('rc_',)

_EXTRACT_CLEAR_KEYS = [
    'excel_result', 'is_region_mode', 'region_results_summary',
    'ref_results_summary', 'terminal_results',
]
_EXTRACT_CLEAR_PREFIXES = ()

_RESTART_CLEAR_KEYS = [
    'excel_result', 'output_filename', 'processing_settings',
    'results', 'is_region_mode', 'region_analyses',
    'region_results_summary', 'ref_results_summary', 'download_done',
    'terminal_results', 'global_region_checked_names',
]
_RESTART_CLEAR_PREFIXES = ('rc_',)


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
    """この図面の領域名選択を集める。「領域一覧」でのグローバル決定（優先）と、
    「領域の確認」での個別選択（グローバルで未決定の領域のみ）を統合する。

    候補1件の領域は「領域一覧」チェックが唯一のゲート（未チェックなら抽出
    しない）。候補0件の領域はここでは何も返さない（呼び出し先の
    build_region_results 等が自動 "no name" 命名を行う）。
    """
    checked_names = set(st.session_state.get('global_region_checked_names', []))
    name_selections = {}
    for reg in analysis.get('regions', []):
        decided = resolve_globally_decided_name(reg, checked_names)
        if decided is not None:
            name_selections[(fname, reg['id'])] = [decided]
            continue
        if not is_region_pending_selection(reg, checked_names):
            continue
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

    ON になったとき、(1) 同一領域の名称候補チェックボックスを全て OFF にする、
    (2) この領域自身の候補テキストと一致するチェックボックスを、他領域
    （候補を1つでも持ち、かつ同一ファイル内で重なりのある領域を除く）でも
    OFF にする、(3) (2) の結果その他領域の候補が全て OFF になった場合は、
    その領域自身の「領域選択しない」も ON にする（2026-07-26 ユーザー指摘）。

    旧実装は (2)(3) を行わず「この領域だけの個別判断」としていた（「選択しない」は
    実在する名称候補ではないため、同名候補を持つ他領域まで一律「選択しない」
    にする理由がない、という理由）。しかし「選択しない」を選んだ以上この領域の
    候補群はどれも採用されなくなるにもかかわらず、`_on_change_radio()` の (2)
    によって以前 ON 側へ同期された他領域の同名チェックボックスがそのまま
    残っていると、「この領域では不採用と決めた名称が、別領域では採用済みの
    まま」という矛盾した状態になる。ON化の伝播（`_on_change_radio`）と対称的に
    OFF化も伝播させることで解消する。ON化と異なり「クリックされた1候補」では
    なく「この領域の候補すべて」が対象になる点に注意（この領域自身がどの候補
    にも紐付かないことを選んだため）。

    **候補1件のみ（自動確定）の領域も対象に含める**: `_on_change_radio()` の
    (2) は「候補が2件以上＝選択肢ありの領域のみ」を同期対象とし、候補1件のみの
    領域（ユーザーが能動的に選んだわけではない自動確定）は除外している。だが
    この OFF 伝播ではその除外を適用しない。実例（2026-07-26 ユーザー報告、
    `DE5027-563-03A.dxf`）: 候補1件のみの領域が複数（領域21-24が候補
    `NXSAF FX1`、領域25-29が候補`SX13 FX`）連続しており、そのうち1つで
    「選択しない」を選んでも、候補1件のみの除外により兄弟領域に伝播せず
    「選択しない」も同期しなかった。ON伝播（能動的な選択と紛れることを防ぐ
    ための除外）と異なり、この OFF 伝播は「今まさに選ばれた選択しないという
    能動的な判断」を伝えるものであり、伝播先が候補1件だったかどうかは無関係
    （むしろ「同じ孤立候補を持つ領域群を一貫させる」というこの機能の目的そのもの
    に合致する）。

    さらに (3): OFF化の結果、他領域の候補が1つも選択されていない状態になると、
    「どの候補も選んでいない」のに「領域選択しない」も OFF という、実質的に
    未決定なまま宙に浮いた状態になる（ユーザー指摘: 「何も選択されていない
    領域は『領域選択しない』を ON にする必要がある」）。この領域自身の候補
    チェックボックスが（今回の伝播とは無関係な既存の OFF 状態も含めて）全て
    OFF になっていれば、その領域の「領域選択しない」を ON にして状態を明確化
    する。
    """
    ck = f"rc_{fname}_{reg_id}_none"
    if not st.session_state.get(ck, False):
        return
    for j in range(n_cands):
        st.session_state[f"rc_{fname}_{reg_id}_{j}"] = False

    analyses = st.session_state.get('region_analyses', {})
    clicked_region = next(
        (r for r in analyses.get(fname, {}).get('regions', []) if r['id'] == reg_id), None)
    if not clicked_region:
        return
    clicked_texts = {t for _d, t in clicked_region.get('name_candidates', [])}
    clicked_polygon = clicked_region['polygon']

    for fn2, an2 in analyses.items():
        for r2 in an2.get('regions', []):
            if fn2 == fname and r2['id'] == reg_id:
                continue
            cands2 = r2.get('name_candidates', [])
            if not any(t2 in clicked_texts for _d2, t2 in cands2):
                continue  # この領域の候補を1つも持たない他領域には影響しない
                          # （候補0件・候補1件のみ＝自動確定の領域も対象に含める。
                          # docstring「候補1件のみ（自動確定）の領域も対象に含める」参照）
            if (fn2 == fname and regions_overlap(clicked_polygon, r2['polygon'])):
                continue  # 重なり（内包/部分重複）のある領域同士は同期しない
            for j2, (_d2, t2) in enumerate(cands2):
                if t2 in clicked_texts:
                    st.session_state[f"rc_{fn2}_{r2['id']}_{j2}"] = False
            if not any(st.session_state.get(f"rc_{fn2}_{r2['id']}_{j2}", False)
                       for j2 in range(len(cands2))):
                st.session_state[f"rc_{fn2}_{r2['id']}_none"] = True


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
            "既定では、図面枠内・図面情報枠（タイトルブロック）外の機器符号"
            "（Reference Designator）パターンに一致するラベルを抽出します。",
            "",
            "**主なオプション：**",
            "- 機器符号（候補）以外も抽出：OFF（既定）は図面枠内・図面情報枠外の"
            "機器符号パターンに一致するラベルを抽出します。ONにすると"
            "図面枠・図面情報枠・パターンの制限なしに全ラベルを抽出します",
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
            "「領域一覧」で抽出したい領域名にチェックを入れます（複数選択可）。"
            "チェックで決定できなかった領域は「領域の確認」で個別に選択します",
            "4. 「ラベルを抽出」ボタンで処理を実行します（結果はそのままExcelとして"
            "ダウンロードできます）。抽出後はオプション等を振り返れますが変更は"
            "できません（変更するには「新しい抽出を開始」を押してください）",
            "",
            "**Excelファイルの内容：**",
            "- Summaryシート：全ファイルの抽出結果概要",
            "- Totalシート：全ファイル横断のラベル集計（領域検出時も出力されます）",
            "- 領域一覧・領域別ラベル一覧シート（領域検出時）：検出した領域ごとの集計",
            "- 各ファイルシート：個別ファイルの抽出ラベル一覧（領域検出時は「領域」列付き）",
            "- TB Listシート（端子一覧を抽出時）：端子台ごとの端子番号一覧",
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
        "DXFファイルを選択してください（複数可）",
        type="dxf",
        accept_multiple_files=True,
        key=f"dxf_uploader_{st.session_state['uploader_version']}",
    )
    return uploaded_files


def _render_options_section(locked):
    """オプション欄（機器符号以外も抽出・図番等抽出・領域検出・端子一覧抽出・
    並び替え・出力ファイル名）を表示し、選択値を dict で返す。

    `locked` が True（抽出結果表示中）の間は全ウィジェットを disabled にし、
    選択内容を振り返れるが変更できない状態にする。"""
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
            disabled=locked,
            help="チェックなし（既定）：図面枠内・図面情報枠（タイトルブロック）外の"
                 "ラベルのうち、機器符号（Reference Designator）パターンに一致する"
                 "ものだけを抽出します。\n\n"
                 "チェックあり：図面枠外・図面情報枠内も含め、すべてのラベルを"
                 "そのまま抽出します（機器符号パターンの判定は行いません）。\n\n"
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
            disabled=locked,
            help="図面番号（例：EE6868-500-01C）、タイトル、サブタイトルをサマリーシートに抽出します。"
        )
        extract_drawing_numbers_option = extract_all_option
        extract_title_option = extract_all_option

        # c) 領域を検出（矩形領域抽出機能全体のON/OFF）
        enable_region_detection = st.checkbox(
            "領域を検出",
            value=False,
            key='enable_region_detection',
            disabled=locked,
            help="ONにすると「領域検出の詳細設定」「領域を検出」ボタン「領域一覧」"
                 "「領域の確認」セクションが表示され、矩形領域付きでラベルを抽出できます。"
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
                keys=['region_analyses', 'global_region_checked_names'],
                prefixes=_REGION_DETECT_CLEAR_PREFIXES)

        # d) 端子一覧を抽出
        extract_terminal_option = st.checkbox(
            "端子一覧を抽出",
            value=False,
            disabled=locked,
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
        'also_extract_non_circuit': also_extract_non_circuit,
        'extract_all_option': extract_all_option,
        'extract_drawing_numbers_option': extract_drawing_numbers_option,
        'extract_title_option': extract_title_option,
        'enable_region_detection': enable_region_detection,
        'extract_terminal_option': extract_terminal_option,
        'sort_value': sort_value,
        'output_filename': output_filename,
    }


def _render_region_config_form(enable_region_detection, locked):
    """「領域検出の詳細設定」フォームを表示し、確定済みの region_cfg を返す
    （「設定完了」が押されるまでは保存済み設定 or 既定値を返す）。

    `locked` が True の間は全フィールド・「設定完了」ボタンを disabled にする。"""
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
                        key='frm_frame_lineweight', disabled=locked)
                    frm_region_lw = st.number_input(
                        "領域境界線の太さ",
                        value=int(saved_cfg['region_lineweight']),
                        step=5,
                        help="矩形領域の境界線の太さ（lineweight）",
                        key='frm_region_lineweight', disabled=locked)
                    frm_cp_margin = st.number_input(
                        "接続点判定マージン（座標）",
                        value=float(saved_cfg['connection_point_margin']),
                        min_value=0.0, step=0.01, format="%.2f",
                        help="接続点（円）が境界線からこの座標距離以内なら「境界上」とみなします。"
                             "縦ギャップ上に接続点がある場合の橋渡し除外にも使用します。",
                        key='frm_connection_point_margin', disabled=locked)
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
                        key='frm_region_color', disabled=locked)
                    frm_filter_prefixes = st.text_input(
                        "抽出したい領域名の先頭文字列（カンマ区切り・複数可）",
                        value=','.join(saved_cfg.get('name_filter_prefixes', ())),
                        help="指定すると、名称候補がこの文字列のいずれかで始まる領域は、"
                             "面積比の閾値を満たさなくても抽出対象になります"
                             "（前方一致。全角/半角は区別しません）。空欄なら従来通り"
                             "面積比のみで判定します。",
                        key='frm_name_filter_prefixes', disabled=locked)
                    frm_terms = st.text_input(
                        "領域名称候補から除外する単語（カンマ区切り）",
                        value=','.join(saved_cfg['name_exclude_terms']),
                        help="これらの語を含むラベルを名称候補から除外します",
                        key='frm_name_exclude_terms', disabled=locked)
                with rc2:
                    frm_area_pct = st.number_input(
                        "最小面積（単独領域・図面枠面積比 %）",
                        value=min(99, max(1, round(saved_cfg['area_ratio'] * 100))),
                        min_value=1, max_value=99, step=1,
                        help="1つの閉領域が単独でこの面積比（四捨五入した整数%）以上のとき抽出対象とします",
                        key='frm_area_ratio_pct', disabled=locked)
                    frm_group_pct = st.number_input(
                        "最小面積（同名複数領域・図面枠面積比 %）",
                        value=min(50, max(1, round(saved_cfg['group_area_ratio'] * 100))),
                        min_value=1, max_value=50, step=1,
                        help="同じ名称の複数ピースを合算したとき、この面積比（四捨五入した整数%）以上なら抽出対象とします。"
                             "第1図面で成立した名称は他図面でも面積不問で抽出します。",
                        key='frm_group_area_ratio_pct', disabled=locked)
                    frm_name_max = st.number_input(
                        "領域名称ラベルの境界線からの最大距離（座標）",
                        value=min(20, max(1, int(saved_cfg.get('name_max_dist', 10.0)))),
                        min_value=1, max_value=20, step=1,
                        help="下端境界線からこの座標距離以内のラベルを名称候補とします",
                        key='frm_name_max_dist', disabled=locked)
                    frm_name_min = st.number_input(
                        "領域名称ラベルの境界線からの最小距離（座標）",
                        value=min(10, max(1, int(saved_cfg['name_min_dist']))),
                        min_value=1, max_value=10, step=1,
                        help="この距離未満（境界線分上）のラベルは名称候補から除外します",
                        key='frm_name_min_dist', disabled=locked)
                    frm_min_letters = st.number_input(
                        "領域名称候補に必要な文字数",
                        value=int(saved_cfg['name_min_letters']),
                        min_value=1, step=1,
                        help="英字がこの文字数以上のラベルのみ名称候補とします",
                        key='frm_name_min_letters', disabled=locked)

                submitted = st.form_submit_button("設定完了", type="primary", disabled=locked)

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
    return region_cfg


def _render_region_detection_trigger(uploaded_files, region_cfg, extract_all_option):
    """「領域を検出」ボタンを表示し、押されたら各ファイルを解析して
    `st.session_state['region_analyses']` に格納する。

    表示するかどうか（領域検出ON かつ未検出時のみ）は呼び出し元の `app()` が
    判定する。表示される場面は常に「これが次に行うべき唯一の操作」であるため、
    常に primary（青）で表示する。"""
    # ============================================================
    # 領域を検出
    # ============================================================
    if st.button("領域を検出", key="detect_regions_btn", type="primary"):
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


def _render_global_region_names_section(locked):
    """「領域一覧」セクション: 検出済みの全ファイル・全領域の名称候補
    （単一候補・複数候補を問わず）を重複なくABC順で一覧表示し、抽出したい
    領域名にチェックを入れさせる（複数選択可）。

    チェックした名称に一致する候補を持つ領域は、その名称で自動的に決定される
    （1領域が複数のチェック済み候補を持つ場合は最も優先度の高い候補を採用）。
    候補が2件以上あり、いずれもチェックされていない領域だけが
    「領域の確認」での個別選択の対象になる。候補1件の領域は、この一覧での
    チェックが抽出するかどうかの唯一のゲートになる（未チェックなら抽出しない）。
    """
    if 'region_analyses' not in st.session_state:
        return
    analyses = st.session_state['region_analyses']
    all_names = collect_unique_region_candidate_names(analyses)
    if not all_names:
        return

    st.subheader("領域一覧")
    st.caption(
        "検出された全ファイル・全領域の名称候補（複数候補を持つ領域も含む）を"
        "重複なく一覧表示しています。抽出したい領域名にチェックを入れてください"
        "（複数選択可）。チェックした名称に一致する候補を持つ領域は、その名称で"
        "自動的に抽出されます。候補が2件以上あり、いずれもチェックされていない"
        "領域のみ、下の「領域の確認」で個別に選択してください。"
        "候補が1件だけの領域は、ここでチェックしない限り抽出されません。"
    )
    if 'global_region_checked_names' not in st.session_state:
        st.session_state['global_region_checked_names'] = []
    st.multiselect(
        "抽出する領域名",
        options=all_names,
        key='global_region_checked_names',
        disabled=locked,
    )


def _render_region_confirmation_section(enable_region_detection, locked):
    """「領域の確認」セクション: 「領域一覧」で決定できなかった領域
    （候補が2件以上あり、いずれもチェックされていないもの）だけを対象に、
    名称候補チェックボックス群を表示する。"""
    # ============================================================
    # 領域の確認
    # ============================================================
    if enable_region_detection and 'region_analyses' in st.session_state:
        analyses = st.session_state['region_analyses']
        checked_names = set(st.session_state.get('global_region_checked_names', []))
        st.subheader("領域の確認")
        st.caption(
            "「領域一覧」のチェックで決定できなかった領域（候補が2件以上あり、"
            "いずれもチェックされていないもの）のみを表示しています。"
        )

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
            pending_regions = [
                r for r in regions if is_region_pending_selection(r, checked_names)
            ]
            st.caption(
                f"図面枠 {n_frames} 個 / 検出領域 {len(regions)} 個"
                f"（うち要選択 {len(pending_regions)} 個）"
            )
            if err:
                st.warning(err)
                continue

            if not regions:
                st.info("面積条件を満たす領域が検出されませんでした。詳細設定の調整をお試しください。")
                continue

            if not pending_regions:
                st.caption("　　（「領域一覧」で決定済み、または候補なしのため個別選択は不要です）")
                continue

            for reg_idx, reg in enumerate(pending_regions):
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
                default_idx = _compute_default_candidate_index(fname, reg, cands, analyses)

                for i, (d, t) in enumerate(cands):
                    ck = f"rc_{fname}_{reg['id']}_{i}"
                    if ck not in st.session_state:
                        st.session_state[ck] = (i == default_idx)
                    st.checkbox(
                        f"　{t}　（境界線からの距離 {d:.0f}）",
                        key=ck,
                        on_change=_on_change_radio,
                        args=(fname, reg['id'], i, len(cands), t),
                        disabled=locked,
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
                    args=(fname, reg['id'], len(cands)),
                    disabled=locked,
                )


def _handle_extract_button(uploaded_files, region_cfg, opts):
    """「ラベルを抽出」ボタンを表示し、押されたらモード（領域付き/通常 ×
    機器符号候補のみ/機器符号以外も抽出）に応じた抽出処理を実行する。

    表示するかどうか（結果表示中でない、かつ領域検出OFFまたは検出済み）は
    呼び出し元の `app()` が判定する。表示される場面は常に「これが次に
    行うべき唯一の操作」であるため、常に primary（青）で表示する。

    `opts` の中身をローカル変数へ展開してから本体（元の app() 内コードと
    完全に同一）を実行する——本体側は変更せず、置換によるキーワード引数名
    （例: `extract_labels(extract_drawing_numbers_option=True, ...)`)との
    衝突リスクを避けるため。"""
    extract_terminal_option = opts['extract_terminal_option']
    also_extract_non_circuit = opts['also_extract_non_circuit']
    sort_value = opts['sort_value']
    extract_drawing_numbers_option = opts['extract_drawing_numbers_option']
    extract_title_option = opts['extract_title_option']
    output_filename = opts['output_filename']
    extract_all_option = opts['extract_all_option']

    # ============================================================
    # ラベルを抽出
    # ============================================================
    if st.button("ラベルを抽出", type="primary"):
        try:
            frame_lineweight = int(region_cfg['frame_lineweight'])
            with st.spinner(f'{len(uploaded_files)}個のDXFファイルを処理中...'):
                # 前回の結果をクリアする
                _clear_session_keys(
                    keys=_EXTRACT_CLEAR_KEYS,
                    prefixes=_EXTRACT_CLEAR_PREFIXES)

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
                        # 機器符号パイプライン: 抽出して直ちにExcelを生成する
                        ref_data_by_file = {}
                        for fname, analysis in analyses.items():
                            uf = next(f for f in uploaded_files if f.name == fname)
                            with _temp_dxf_file(uf) as tmp:
                                data = ref_designator.extract_ref_designator_data(
                                    tmp, frame_lineweight=frame_lineweight, original_filename=fname)
                            data['main_drawing_number'] = analysis.get('main_drawing_number')
                            data['title'] = analysis.get('title')
                            data['subtitle'] = analysis.get('subtitle')
                            ref_data_by_file[fname] = data

                        name_selections_by_file = {
                            fname: _gather_name_selections(fname, analyses[fname])
                            for fname in ref_data_by_file
                        }
                        region_results = ref_designator.build_ref_designator_region_results(
                            ref_data_by_file, analyses, name_selections_by_file, sort_value)
                        st.session_state.excel_result = create_region_excel_output(
                            region_results, terminal_results=terminal_results)
                        st.session_state.output_filename = output_filename
                        st.session_state['is_region_mode'] = True
                        st.session_state['region_results_summary'] = {
                            f: {k: v for k, v in d.items() if k not in ('named', 'rows')}
                            for f, d in region_results.items()
                        }
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
                        # 機器符号パイプライン: 抽出して直ちにExcelを生成する
                        ref_data_by_file = {}
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
                            ref_data_by_file[uploaded_file.name] = data

                        ref_final = ref_designator.build_ref_designator_final(ref_data_by_file, sort_value)
                        st.session_state.excel_result = create_ref_designator_excel_output(
                            ref_final, sort_value, terminal_results=terminal_results)
                        st.session_state.output_filename = output_filename
                        st.session_state['is_region_mode'] = False
                        st.session_state['ref_results_summary'] = ref_final
                        _set_processing_settings(False, extract_drawing_numbers_option, extract_title_option)

                st.session_state['download_done'] = False

            st.rerun()
        except Exception as e:
            handle_error(e)


def _render_results_section():
    """抽出結果（統計・ダウンロードボタン・再抽出ボタン）を表示する。"""
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
            st.success(f"{len(summ)}個のDXFファイルから機器符号の抽出が完了しました")
        else:
            results = st.session_state.get('results', {})
            st.success(f"{len(results)}個のDXFファイルからラベル抽出が完了しました")

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
                        '機器符号数': sum(r['個数'] for r in d.get('rows', [])),
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


def app():
    """全体の表示制御（4状態）:

    A) 結果表示中（`excel_result`あり）: オプション・詳細設定・領域一覧・
       領域の確認は表示するが操作不可（disabled、振り返り専用）。
       「領域を検出」「ラベルを抽出」ボタンはいずれも非表示
       （「新しい抽出を開始」でのみ次のサイクルへ進める）。
    B) 領域検出ON・未検出: 「領域を検出」ボタンのみ表示（常に青）。
       「ラベルを抽出」・領域一覧・領域の確認は非表示（region_analysesが無いため）。
    C) 領域検出ON・検出済み: 「領域を検出」は非表示（役目を終えたため）、
       「ラベルを抽出」のみ表示（常に青）。領域一覧・領域の確認を表示。
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

    region_cfg = _render_region_config_form(opts['enable_region_detection'], locked)

    detected = 'region_analyses' in st.session_state

    if not locked and opts['enable_region_detection'] and not detected:
        _render_region_detection_trigger(
            uploaded_files, region_cfg, opts['extract_all_option'])

    _render_global_region_names_section(locked)

    _render_region_confirmation_section(opts['enable_region_detection'], locked)

    if not locked and (not opts['enable_region_detection'] or detected):
        _handle_extract_button(uploaded_files, region_cfg, opts)

    _render_results_section()


if __name__ == "__main__":
    app()
