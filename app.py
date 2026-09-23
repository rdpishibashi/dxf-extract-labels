import streamlit as st
import os
import sys
import contextlib
import time
import traceback
from io import BytesIO
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model')
sys.path.insert(0, model_path)

from model.extract_labels import extract_labels
from model.region_detector import (
    analyze_dxf_regions, DEFAULT_REGION_CONFIG, regions_overlap,
    collect_unique_region_candidate_names, resolve_globally_decided_name,
    is_region_pending_selection, pending_candidates_for_region,
)
from model.excel_output import (
    create_ref_designator_excel_output, create_region_excel_output,
)
from model.common_utils import save_uploadedfile
from model import ref_designator
from model import terminal_detector
from model import assembly_compare
from config import region_detection_config


def handle_error(e, show_traceback=True):
    """エラーを適切に処理して表示する"""
    st.error(f"エラーが発生しました: {str(e)}")
    if show_traceback:
        st.error(traceback.format_exc())

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
_FRAME_NOT_FOUND_ERROR_MARKER = '領域探索を実施することができませんでした'

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
_REGION_DETECT_CLEAR_PREFIXES = ('rc_', 'grc_')

_EXTRACT_CLEAR_KEYS = [
    'excel_result', 'is_region_mode', 'region_results_summary',
    'ref_results_summary', 'terminal_results', 'extract_preview_expanded',
] + _ASSEMBLY_COMPARE_CLEAR_KEYS
_EXTRACT_CLEAR_PREFIXES = ()

_RESTART_CLEAR_KEYS = [
    'excel_result', 'output_filename', 'is_region_mode', 'region_analyses',
    'region_results_summary', 'ref_results_summary', 'download_done',
    'terminal_results', 'region_selection_confirmed', 'extract_preview_expanded',
] + _ASSEMBLY_COMPARE_CLEAR_KEYS
_RESTART_CLEAR_PREFIXES = ('rc_', 'grc_')


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


def _global_checked_region_names():
    """「領域一覧」のチェックボックス（`grc_<name>`）からチェック済みの
    領域名の集合を返す。個々のチェックボックスは `_render_global_region_names_section()`
    が描画する（`st.multiselect` ではなく個別チェックボックスにしたのは、
    ポップアップに隠れず全候補を一覧できるようにするため、2026-07-30）。"""
    prefix = 'grc_'
    return {
        k[len(prefix):] for k, v in st.session_state.items()
        if isinstance(k, str) and k.startswith(prefix) and v
    }


def _gather_name_selections(fname, analysis):
    """この図面の領域名選択を集める。「領域一覧」でのグローバル決定（優先）と、
    「領域の確認」での個別選択（チェック済み候補が2件以上ある領域のみ）を統合する。

    「領域一覧」でチェックされなかった候補は「採用しない」という意思表示と
    みなす（2026-07-30 ユーザー指定）。そのため、候補が0件チェック済みの
    領域はここでは何も返さない（無所属のまま。候補が元々0件の無名領域は
    呼び出し先の `ref_designator.build_named_regions()` 等が自動 "no name"
    命名を行う）。
    """
    checked_names = _global_checked_region_names()
    name_selections = {}
    for reg in analysis.get('regions', []):
        decided = resolve_globally_decided_name(reg, checked_names)
        if decided is not None:
            name_selections[(fname, reg['id'])] = [decided]
            continue
        if not is_region_pending_selection(reg, checked_names):
            continue
        pending_cands = pending_candidates_for_region(reg, checked_names)
        chosen = []
        for i, (_d, t) in enumerate(pending_cands):
            if st.session_state.get(f"rc_{fname}_{reg['id']}_{i}"):
                chosen.append(t)
        if chosen:
            name_selections[(fname, reg['id'])] = chosen
    return name_selections


def _on_change_radio(fname, reg_id, clicked_idx, n_cands, clicked_text, checked_names):
    """領域名チェックボックスをラジオボタン的に動作させるコールバック。

    `checked_names` は「領域一覧」でのチェック済み名称集合。他領域の候補は
    `pending_candidates_for_region()` で同じ集合に絞り込んでから走査する
    （`rc_<fname>_<reg_id>_<i>` のインデックスが実際の描画と対応するように
    するため、2026-07-30）。

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
            cands2 = pending_candidates_for_region(r2, checked_names)
            if len(cands2) <= 1:
                continue  # 選択肢なし（自動確定）の領域は対象外
            if not any(t2 == clicked_text for _d2, t2 in cands2):
                continue  # この名称を候補に持たない領域には影響しない
            if (fn2 == fname and clicked_polygon
                    and regions_overlap(clicked_polygon, r2['polygon'])):
                continue  # 重なり（内包/部分重複）のある領域同士は同期しない
            for j2, (_d2, t2) in enumerate(cands2):
                st.session_state[f"rc_{fn2}_{r2['id']}_{j2}"] = (t2 == clicked_text)


def _on_change_none(fname, reg_id, n_cands, checked_names):
    """「領域選択しない」チェックボックスをラジオボタン的に動作させるコールバック。

    `checked_names` は「領域一覧」でのチェック済み名称集合（他領域の候補の
    絞り込みに使う。理由は `_on_change_radio()` の docstring 参照）。

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
    clicked_texts = {t for _d, t in pending_candidates_for_region(clicked_region, checked_names)}
    clicked_polygon = clicked_region['polygon']

    for fn2, an2 in analyses.items():
        for r2 in an2.get('regions', []):
            if fn2 == fname and r2['id'] == reg_id:
                continue
            cands2 = pending_candidates_for_region(r2, checked_names)
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


def _compute_default_candidate_index(fname, reg, cands, analyses, checked_names):
    """他の図面/領域で選択済みの名称があれば、それに一致する候補のインデックスを
    デフォルトとして返す（無ければ0）。

    `cands` は呼び出し元がこの領域用に描画する候補リスト（`pending_candidates_for_region()`
    で「領域一覧」チェック済みのものに絞り込み済み）。他領域を走査する際も
    同じ絞り込みを適用しないと、`rc_<fname>_<reg_id>_<i>` のインデックス `i` が
    実際に描画されたチェックボックスと対応しなくなる（2026-07-30）。

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
            cands2 = pending_candidates_for_region(r2, checked_names)
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
            "「領域一覧」で抽出したい領域名にチェックを入れます（複数選択可）。"
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
            help="ONにすると「領域を検出」ボタン「領域一覧」「領域の確認」"
                 "セクションが表示され、矩形領域付きでラベルを抽出できます。"
                 "OFFにすると通常モードになります。"
                 "（検出パラメータは config.py の region_detection_config で設定します）"
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
            help="タイトルが「UNIT内結線図」の図面を対象に、TBで始まり英大文字・数字が"
                 "続く端子台ラベルに対応する矩形（LINE+CIRCLEで構成）を検出し、矩形内の"
                 "端子番号を「TB List」シート（抽出ラベル一覧シート直後）に出力します。"
                 "「端子台」でユニークをとり、同じ番号が複数ある場合は件数をカッコで"
                 "表示します（例: 7(2)）。\n\n"
                 "サブタイトルが「TB COMPONENT」の図面（端子台の全端子番号一覧ページ）"
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
                 "比較シート無しでラベル抽出自体は完了します。"
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
    """`config.py` の `region_detection_config`（旧「領域検出の詳細設定」フォームの
    11設定）から `region_cfg` を組み立てる。

    2026-09-10、UIフォーム（st.expander + st.form、「設定完了」ボタン）を撤去し
    config.py 編集方式に統一した（ユーザー指示）。UIを介さないため戻り値は常に
    「確定済み」の設定であり、旧実装にあった「設定完了」ボタンを押すまでは
    保存済み設定 or 既定値を返す、という2段階の状態は無くなった。

    組み立て順: `DEFAULT_REGION_CONFIG`（アルゴリズム内部の既定値一式）→
    `region_detection_config` の11設定で上書き → `region_detection_config.
    ADVANCED_OVERRIDES` でさらに上書き（`DEFAULT_REGION_CONFIG` のキーなら
    何でも指定可能）。既定値のみの場合、返り値は `DEFAULT_REGION_CONFIG` と
    完全に等価になる（`tests/regression/test_config_region_cfg.py` で固定）。
    """
    cfg = dict(DEFAULT_REGION_CONFIG)
    rc = region_detection_config
    cfg.update({
        'frame_lineweight': rc.FRAME_LINEWEIGHT,
        'region_lineweight': rc.REGION_LINEWEIGHT,
        'connection_point_margin': rc.CONNECTION_POINT_MARGIN,
        'region_color': rc.REGION_COLOR,
        'name_filter_prefixes': tuple(rc.NAME_FILTER_PREFIXES),
        'name_exclude_terms': tuple(rc.NAME_EXCLUDE_TERMS),
        'area_ratio': rc.AREA_RATIO_PERCENT / 100.0,
        'group_area_ratio': rc.GROUP_AREA_RATIO_PERCENT / 100.0,
        'name_max_dist': float(rc.NAME_MAX_DIST),
        'name_min_dist': float(rc.NAME_MIN_DIST),
        'name_min_letters': rc.NAME_MIN_LETTERS,
    })
    cfg.update(rc.ADVANCED_OVERRIDES)
    return cfg


def _render_region_detection_trigger(uploaded_files, region_cfg):
    """「領域を検出」ボタンを表示し、押されたら各ファイルを解析して
    `st.session_state['region_analyses']` に格納する。

    表示するかどうか（領域検出ON かつ未検出時のみ）は呼び出し元の `app()` が
    判定する。表示される場面は常に「これが次に行うべき唯一の操作」であるため、
    常に primary（青）で表示する。

    図番・タイトル・サブタイトルは v3.0.0で常に抽出する（旧「図面番号・タイトル・
    サブタイトルを抽出」オプション廃止に伴う）。`extract_labels()` を呼ぶ際は
    `extract_drawing_numbers_option`/`extract_title_option` を必ずセットで渡す
    （`Tools/CLAUDE.md` の呼び出し側オプション整合性ルール）。"""
    # ============================================================
    # 領域を検出
    # ============================================================
    if st.button("領域を検出", key="detect_regions_btn", type="primary"):
        try:
            with st.spinner('図面枠と矩形領域を検出中...'):
                # 進捗表示（ファイル名・件数・経過秒数）は spinner の下に出す
                # ため、st.empty() は spinner の内側（2026-09-23 ユーザー指示、
                # DXF-diff-managerと同じ書式。`_handle_extract_button()` 参照）。
                status_placeholder = st.empty()
                start_time = time.time()
                analyses = {}
                total_files = len(uploaded_files)
                for idx, uf in enumerate(uploaded_files, start=1):
                    elapsed = time.time() - start_time
                    status_placeholder.text(
                        f"領域を検出中: {uf.name}（{idx}/{total_files}件、経過 {elapsed:.1f} 秒）")
                    with _temp_dxf_file(uf) as tmp:
                        analysis = analyze_dxf_regions(tmp, region_cfg)
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
                status_placeholder.empty()
                st.session_state['region_analyses'] = analyses
                _clear_session_keys(
                    keys=_REGION_DETECT_CLEAR_KEYS,
                    prefixes=_REGION_DETECT_CLEAR_PREFIXES)
            st.rerun()
        except Exception as e:
            handle_error(e)


def _invalidate_region_selection_confirmation():
    """「領域一覧」のチェック状態が変更されたら「領域選択を完了」を取り消す
    コールバック。再度「領域選択を完了」を押すまで「領域の確認」「ラベルを
    抽出」を隠す（決定内容が変わった以上、再確認が必要なため）。"""
    st.session_state['region_selection_confirmed'] = False


def _render_global_region_names_section(locked):
    """「領域一覧」セクション: 検出済みの全ファイル・全領域の名称候補
    （単一候補・複数候補を問わず）を重複なくABC順で一覧表示し、抽出したい
    領域名にチェックを入れさせる（複数選択可）。「領域選択を完了」ボタンを
    押すまでは「領域の確認」「ラベルを抽出」は表示されない（`app()` 側が
    `region_selection_confirmed` を見て判定する）。

    チェック済み候補が「ちょうど1件」の領域は、その名称で自動的に決定される。
    チェック済み候補が「2件以上」残る領域だけが「領域の確認」での個別選択の
    対象になる（どれを採用するかは候補どうしが競合しているため個別に選ばせる）。
    チェック済み候補が「0件」の領域は、自動的に除外され「領域の確認」にも
    表示されない（未チェック＝抽出しないという意思表示のため）。候補1件の
    領域も同様に、この一覧でのチェックが抽出するかどうかの唯一のゲートになる
    （2026-07-30 ユーザー指定で確定した仕様）。
    """
    if 'region_analyses' not in st.session_state:
        return
    analyses = st.session_state['region_analyses']
    all_names = collect_unique_region_candidate_names(analyses)
    if not all_names:
        # 「領域を検出」は実行済みだが、面積条件（config.py の area_ratio /
        # group_area_ratio）を満たす領域が1件も無かった（`analyses[...]['regions']`
        # が全ファイルで空）ケース。旧UIの「面積条件を満たす領域が検出されません
        # でした。詳細設定の調整をお試しください。」から、詳細設定UIが撤去された
        # ことに伴い2026-09-10に文言変更（ユーザー指示）。
        st.info("一定面積の領域は検出されませんでした。")
        return

    st.subheader("領域一覧")
    st.caption(
        "検出された全ファイル・全領域の名称候補（複数候補を持つ領域も含む）を"
        "重複なく一覧表示しています。抽出したい領域名にチェックを入れてください"
        "（複数選択可）。ある領域のチェック済み候補がちょうど1件なら、その名称で"
        "自動的に抽出されます。チェック済み候補が2件以上残る領域のみ、"
        "「領域選択を完了」を押した後に表示される「領域の確認」でどれを採用するか"
        "個別に選択してください。候補を1つもチェックしなかった領域は抽出されません"
        "（「領域の確認」にも表示されません）。"
    )
    # ポップアップに隠さず全候補を一覧できるよう、st.multiselect ではなく
    # チェックボックスを使う。st.container(horizontal=True) はflexboxで
    # 子要素（チェックボックスは既定 width='content' で自身のラベル幅にしか
    # 広がらない）を横に並べ、ブラウザー幅に収まらない分は自動的に折り返す
    # ため、指定した列数無しでもブラウザー幅に応じた2〜4段組相当のレスポンシブ
    # 表示になる（`tests/unit/test_terminal_detector.py` 等で使う固定列
    # `st.columns()` と異なり、幅に応じた折り返し数の変化に追従する）。
    with st.container(horizontal=True):
        for name in all_names:
            ck = f"grc_{name}"
            if ck not in st.session_state:
                st.session_state[ck] = False
            st.checkbox(
                name, key=ck, disabled=locked,
                on_change=_invalidate_region_selection_confirmation,
            )

    # 「領域選択を完了」: 確定済みなら非表示にする（「領域を検出」ボタンと同じ
    # パターン——役目を終えたら消え、チェックを変更すると
    # `_invalidate_region_selection_confirmation` により再度必要になる）。
    if not st.session_state.get('region_selection_confirmed'):
        if st.button("領域選択を完了", type="primary"):
            st.session_state['region_selection_confirmed'] = True
            st.rerun()


def _region_confirmation_targets(analyses, checked_names):
    """「領域の確認」に表示すべきファイルを判定する（純粋関数）。

    表示対象は次のいずれか:
      - 「領域一覧」でチェック済みの候補が2件以上残っている領域（＝どれを
        採用するか競合している領域）を1つ以上持つファイル
      - 図面枠未検出以外の genuine なエラーを持つファイル（要選択領域が
        0件でも、エラー自体をユーザーに見せる必要があるため対象に含める）

    図面枠が見つからない既知の未対応ケース（`_FRAME_NOT_FOUND_ERROR_MARKER`）
    は対象から除く。戻り値はファイル名でソートした
    `[(fname, analysis, pending_regions), ...]`。
    """
    targets = []
    for fname, analysis in sorted(analyses.items()):
        err = analysis.get('error')
        if err and _FRAME_NOT_FOUND_ERROR_MARKER in err:
            continue  # 図面枠が見つからない既知の未対応ケースは表示しない
        regions = analysis.get('regions', [])
        pending_regions = [
            r for r in regions if is_region_pending_selection(r, checked_names)
        ]
        if not err and not pending_regions:
            continue  # 個別選択が不要なファイルは表示しない
        targets.append((fname, analysis, pending_regions))
    return targets


def _render_region_confirmation_section(enable_region_detection, locked):
    """「領域の確認」セクション: 「領域一覧」でチェック済みの候補が2件以上
    残っている領域（＝どれを採用するか競合している領域）だけを対象に、
    チェック済み候補のみの名称候補チェックボックス群を表示する。チェック済み
    候補が0件の領域は自動的に除外され、ここには一切表示されない
    （2026-07-30 ユーザー指定で確定した仕様）。個別選択が不要なファイル
    （エラーが無く、かつ要選択の領域が1つも無いファイル）はセクション自体に
    表示しない（2026-07-30 ユーザー指定）。

    「領域選択を完了」が押される（`region_selection_confirmed` が True になる）
    までは表示しない（2026-07-30 ユーザー指定。「領域一覧」でのチェック中に
    表示内容が頻繁に変化するのを避け、ユーザーが一括選択を終えたタイミングで
    まとめて残りの個別選択を提示する）。

    表示対象ファイルが1件も無い場合（＝全領域が「領域一覧」で決定済み）は、
    見出し・キャプションを含めセクション自体を一切描画しない（2026-09-10
    ユーザー指定。要選択領域の名称決定・「（領域選択しない）」・genuine な
    エラー表示は、このセクションにしか操作手段が無い一方、それが1つも無い
    「空」の状態では何のアクションも取れないため）。"""
    # ============================================================
    # 領域の確認
    # ============================================================
    if not (enable_region_detection and 'region_analyses' in st.session_state
            and st.session_state.get('region_selection_confirmed')):
        return

    analyses = st.session_state['region_analyses']
    checked_names = _global_checked_region_names()
    targets = _region_confirmation_targets(analyses, checked_names)
    if not targets:
        return

    st.subheader("領域の確認")
    st.caption(
        "「領域一覧」でチェックした候補が2件以上残っている領域（どれを採用する"
        "か決まっていないもの）があるファイルのみを表示しています。候補を1つも"
        "チェックしなかった領域は抽出対象から除外され、ここには表示されません。"
    )

    rendered_any = False
    for fname, analysis, pending_regions in targets:
        err = analysis.get('error')
        if rendered_any:
            st.divider()
        rendered_any = True

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
        n_frames = len(analysis.get('frames', []))
        n_regions = len(analysis.get('regions', []))
        st.caption(
            f"図面枠 {n_frames} 個 / 検出領域 {n_regions} 個"
            f"（うち要選択 {len(pending_regions)} 個）"
        )
        if err:
            st.warning(err)
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

                # 「領域一覧」でチェック済みの候補のみを選択肢にする
                # （未チェックの候補は「採用しない」という意思表示とみなし、
                # 選択肢自体に出さない。2026-07-30 ユーザー指定）。
                cands = pending_candidates_for_region(reg, checked_names)
                default_idx = _compute_default_candidate_index(
                    fname, reg, cands, analyses, checked_names)

                for i, (d, t) in enumerate(cands):
                    ck = f"rc_{fname}_{reg['id']}_{i}"
                    if ck not in st.session_state:
                        st.session_state[ck] = (i == default_idx)
                    st.checkbox(
                        f"　{t}　（境界線からの距離 {d:.0f}）",
                        key=ck,
                        on_change=_on_change_radio,
                        args=(fname, reg['id'], i, len(cands), t, checked_names),
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
                    args=(fname, reg['id'], len(cands), checked_names),
                    disabled=locked,
                )


def _resolve_assembly_compare(results, compare_assembly_option):
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
        results, validation['wiring_files'], validation['b_fname'])


def _handle_extract_button(uploaded_files, region_cfg, opts):
    """「ラベルを抽出」ボタンを表示し、押されたらモード（領域付き/通常）に
    応じた抽出処理を実行する。

    表示するかどうか（結果表示中でない、かつ領域検出OFFまたは検出済み）は
    呼び出し元の `app()` が判定する。表示される場面は常に「これが次に
    行うべき唯一の操作」であるため、常に primary（青）で表示する。

    v3.0.0で「機器符号（候補）以外も抽出」「図面番号・タイトル・サブタイトルを
    抽出」の2オプションを廃止したため、モードは領域付き/通常の2つのみ
    （旧4分岐から単純化）。図面枠内・図面情報枠外の全ラベルを常に抽出し
    （機器符号候補かどうかはラベルごとの列として出力）、図番・タイトル・
    サブタイトルも常に抽出する。`extract_labels()` を呼ぶ際は
    `extract_drawing_numbers_option`/`extract_title_option` を必ずセットで渡す
    （`Tools/CLAUDE.md` の呼び出し側オプション整合性ルール）。"""
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
                status_placeholder = st.empty()
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
                    ref_data_by_file = {}
                    total_files = len(analyses)
                    for idx, (fname, analysis) in enumerate(analyses.items(), start=1):
                        elapsed = time.time() - start_time
                        status_placeholder.text(
                            f"ラベルを抽出中: {fname}（{idx}/{total_files}件、経過 {elapsed:.1f} 秒）")
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
                    assembly_result = _resolve_assembly_compare(
                        region_results, compare_assembly_option)
                    st.session_state.excel_result = create_region_excel_output(
                        region_results, terminal_results=terminal_results,
                        assembly_compare=assembly_result)
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
                    ref_data_by_file = {}
                    total_files = len(uploaded_files)
                    for idx, uploaded_file in enumerate(uploaded_files, start=1):
                        elapsed = time.time() - start_time
                        status_placeholder.text(
                            f"ラベルを抽出中: {uploaded_file.name}"
                            f"（{idx}/{total_files}件、経過 {elapsed:.1f} 秒）")
                        with _temp_dxf_file(uploaded_file) as tmp:
                            data = ref_designator.extract_ref_designator_data(
                                tmp, frame_lineweight=frame_lineweight,
                                original_filename=uploaded_file.name)
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
                    assembly_result = _resolve_assembly_compare(ref_final, compare_assembly_option)
                    st.session_state.excel_result = create_ref_designator_excel_output(
                        ref_final, sort_value, terminal_results=terminal_results,
                        assembly_compare=assembly_result)
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


def _apply_type_judgment(selection, validation):
    """「判定を実行」ボタンの処理本体: 確定した判定（`validation`）で比較を
    実行し、Excelを再生成する（`extract_results_full`/`extract_sort_value` に
    保持した抽出結果一式を使い、`terminal_results`・`output_filename` も前回と
    同じ値を引き継ぐ。表4「呼び出し元×Excel出力関数」参照——初回抽出時と
    同じ引数一式を渡すことを確認済み）。呼び出し前に `validation['errors']`
    が空であることを呼び出し元が確認していること。"""
    results = st.session_state['extract_results_full']
    compare_result = assembly_compare.compare_assembly(
        results, validation['wiring_files'], validation['b_fname'])

    terminal_results = st.session_state.get('terminal_results')
    if st.session_state.get('is_region_mode', False):
        st.session_state.excel_result = create_region_excel_output(
            results, terminal_results=terminal_results, assembly_compare=compare_result)
    else:
        st.session_state.excel_result = create_ref_designator_excel_output(
            results, st.session_state['extract_sort_value'],
            terminal_results=terminal_results, assembly_compare=compare_result)

    st.session_state['type_judgment_applied'] = dict(selection)
    st.session_state['download_done'] = False
    st.rerun()


def _render_type_judgment_section():
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
        _apply_type_judgment(selection, validation)


def _render_results_section():
    """抽出結果（統計・ダウンロードボタン・再抽出ボタン）を表示する。

    v3.0.0で「機器符号（候補）以外も抽出」「図面番号・タイトル・サブタイトルを
    抽出」の2オプションを廃止したため、モードは領域付き/通常の2つのみ
    （旧3分岐から単純化）。「領域の確認」セクションを空のときは表示しなく
    したこと（2026-09-10）に伴い、「抽出結果プレビュー」等の置き場として
    「抽出結果」という見出しのセクションをここに新設した。

    「結線図-組立図を比較」関連の判定表示・操作（v3.7.0で「図面種別判定結果」
    セクションへ分離）はこの関数の対象外——呼び出し元 `app()` が
    `_render_type_judgment_section()` をこの関数の直前に呼ぶ。"""
    # ============================================================
    # 抽出結果
    # ============================================================
    if st.session_state.get('excel_result'):
        st.subheader("抽出結果")
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
            )
            st.dataframe(
                preview_xl.parse(preview_sheet), width='stretch', hide_index=True)

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

    detected = 'region_analyses' in st.session_state
    region_selection_confirmed = bool(st.session_state.get('region_selection_confirmed'))

    if not locked and opts['enable_region_detection'] and not detected:
        _render_region_detection_trigger(uploaded_files, region_cfg)

    _render_global_region_names_section(locked)

    _render_region_confirmation_section(opts['enable_region_detection'], locked)

    if not locked and (
        not opts['enable_region_detection']
        or (detected and region_selection_confirmed)
    ):
        _handle_extract_button(uploaded_files, region_cfg, opts)

    _render_type_judgment_section()
    _render_results_section()


if __name__ == "__main__":
    app()
