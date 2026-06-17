import streamlit as st
import os
import tempfile
import sys
from pathlib import Path
import pandas as pd
from io import BytesIO

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, 'utils')
sys.path.insert(0, utils_path)

from utils.extract_labels import (
    extract_labels, get_layers_from_dxf, process_multiple_dxf_files,
)
from utils.region_detector import analyze_dxf_regions, DEFAULT_REGION_CONFIG
from utils.excel_output import create_excel_output, create_region_excel_output, build_region_results
from utils.common_utils import save_uploadedfile, handle_error

st.set_page_config(
    page_title="DXF Extract Labels",
    page_icon="📝",
    layout="wide",
)


@st.cache_data
def _get_layers_cached(file_bytes: bytes) -> list:
    """DXFファイルのレイヤー一覧を取得してキャッシュする（同一ファイルは再処理しない）。"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as f:
        f.write(file_bytes)
        tmp_path = f.name
    try:
        return get_layers_from_dxf(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _on_change_radio(fname, reg_id, clicked_idx, n_cands):
    """領域名チェックボックスをラジオボタン的に動作させるコールバック。
    チェックが ON になったとき、同一領域の他チェックボックスを OFF にする。"""
    ck = f"rc_{fname}_{reg_id}_{clicked_idx}"
    if st.session_state.get(ck, False):
        for j in range(n_cands):
            if j != clicked_idx:
                st.session_state[f"rc_{fname}_{reg_id}_{j}"] = False


def app():
    st.title('DXF Extract Labels')
    st.write('DXFファイルからテキストラベルを抽出し、Excel形式で出力します。')

    with st.expander("ℹ️ プログラム説明", expanded=False):
        st.info("\n".join([
            "このツールは、DXFファイルからテキスト要素（ラベル）を抽出し、Excelファイルに出力します。",
            "",
            "**使用手順：**",
            "1. DXFファイルをアップロードしてください（複数可）",
            "2. 必要に応じてオプションを設定します",
            "3. 「領域を検出」で矩形領域を検出し、各領域の名称を確認・選択します",
            "4. 「ラベルを抽出」ボタンで処理を実行します",
            "",
            "**Excelファイルの内容：**",
            "- Summaryシート：全ファイルの抽出結果概要",
            "- 各ファイルシート：個別ファイルの抽出ラベル一覧（領域検出時は「領域」列付き）",
            "- 領域一覧シート（領域検出時）：検出領域の一覧",
            "- Invalidシート（機器符号妥当性チェック有効時）：適合しない機器符号のリスト",
        ]))

    # ============================================================
    # ファイルアップロード
    # ============================================================
    st.subheader("DXFファイルのアップロード")
    uploaded_files = st.file_uploader(
        "DXFファイルを選択してください（複数可）",
        type="dxf",
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("DXFファイルをアップロードしてください")
        return

    st.success(f"{len(uploaded_files)}個のファイルが選択されました")

    # レイヤー一覧はキャッシュして再処理を防ぐ（ウィジェット操作ごとの遅延を解消）
    available_layers = _get_layers_cached(bytes(uploaded_files[0].getbuffer()))

    # ============================================================
    # オプション
    # ============================================================
    st.subheader("オプション")
    col_left, col_right = st.columns(2)

    with col_left:
        # a) レイヤー選択
        layer_selection_enabled = st.checkbox(
            "特定のレイヤーのみを処理する",
            value=False,
            help="チェックを入れると、選択したレイヤーのみを処理対象とします"
        )
        selected_layers = None
        if layer_selection_enabled and available_layers:
            selected_layers = st.multiselect(
                "処理対象とするレイヤーを選択してください",
                options=available_layers,
                default=available_layers,
                help="複数選択可能です。デフォルトでは全レイヤーが選択されています。"
            )
            if selected_layers:
                st.info(f"{len(selected_layers)}個のレイヤーが選択されています")

        # b) 機器符号以外も抽出（デフォルト OFF = 機器符号のみ）
        also_extract_non_circuit = st.checkbox(
            "機器符号（候補）以外も抽出",
            value=False,
            help="チェックなし（既定）：機器符号パターンに一致するラベルのみ抽出します。\n"
                 "チェックあり：機器符号以外のラベルも含めてすべて抽出します。\n\n"
                 "【機器符号パターン例】\n"
                 "• 英文字のみ: CNCNT, FB\n"
                 "• 英文字+数字: R10, CN3, PSW1\n"
                 "• 英文字+数字+英文字: X14A, RMSS2A\n"
                 "• 括弧付き: FB(), MSS(MOTOR), R10(2.2K)"
        )
        filter_non_parts = not also_extract_non_circuit

        # c) 機器符号妥当性チェック（機器符号のみモードのときのみ表示）
        validate_ref_designators = False
        if not also_extract_non_circuit:
            validate_ref_designators = st.checkbox(
                "機器符号妥当性チェック",
                value=False,
                help="抽出された機器符号がフォーマットに適合するかチェックします。\n"
                     "適合しない機器符号のリストを別シートに出力します。\n"
                     "（例：CBnnn, ELB(CB)nnn, R, Annn等の標準フォーマット）"
            )

        # d) 図面番号・タイトル・サブタイトルを抽出（統合）
        extract_all_option = st.checkbox(
            "図面番号・タイトル・サブタイトルを抽出",
            value=False,
            help="図面番号（例：EE6868-500-01C）、タイトル、サブタイトルをサマリーシートに抽出します。"
        )
        extract_drawing_numbers_option = extract_all_option
        extract_title_option = extract_all_option

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
                frm_region_color = st.number_input(
                    "領域境界線の色(ACI)",
                    value=int(saved_cfg['region_color']),
                    min_value=1, max_value=256, step=1,
                    help="矩形領域の境界線の ACI 色番号（2 = 黄色）",
                    key='frm_region_color')
                frm_cp_margin = st.number_input(
                    "接続点判定マージン（座標）",
                    value=float(saved_cfg['connection_point_margin']),
                    min_value=0.0, step=0.01, format="%.2f",
                    help="接続点（円）が境界線からこの座標距離以内なら「境界上」とみなします。"
                         "縦ギャップ上に接続点がある場合の橋渡し除外にも使用します。",
                    key='frm_connection_point_margin')
            with rc2:
                frm_area_pct = st.number_input(
                    "最小面積（単独領域・図面枠面積比 %）",
                    value=int(saved_cfg['area_ratio'] * 100),
                    min_value=1, max_value=100, step=5,
                    help="1つの閉領域が単独でこの面積比以上のとき抽出対象とします",
                    key='frm_area_ratio_pct')
                frm_group_pct = st.number_input(
                    "最小面積（同名複数領域・図面枠面積比 %）",
                    value=int(saved_cfg['group_area_ratio'] * 100),
                    min_value=1, max_value=100, step=5,
                    help="同じ名称の複数ピースを合算したとき、この面積比以上なら抽出対象とします。"
                         "第1図面で成立した名称は他図面でも面積不問で抽出します。",
                    key='frm_group_area_ratio_pct')
                frm_name_max = st.number_input(
                    "領域名称ラベルの境界線からの最大距離（座標）",
                    value=int(saved_cfg.get('name_max_dist', 10.0)),
                    min_value=0, step=1,
                    help="下端境界線からこの座標距離以内のラベルを名称候補とします",
                    key='frm_name_max_dist')
                frm_name_min = st.number_input(
                    "領域名称ラベルの境界線からの最小距離（座標）",
                    value=int(saved_cfg['name_min_dist']),
                    min_value=0, step=1,
                    help="この距離未満（境界線分上）のラベルは名称候補から除外します",
                    key='frm_name_min_dist')
                frm_min_letters = st.number_input(
                    "領域名称候補に必要な文字数",
                    value=int(saved_cfg['name_min_letters']),
                    min_value=1, step=1,
                    help="英字がこの文字数以上のラベルのみ名称候補とします",
                    key='frm_name_min_letters')
                frm_terms = st.text_input(
                    "領域名称候補から除外する単語（カンマ区切り）",
                    value=','.join(saved_cfg['name_exclude_terms']),
                    help="これらの語を含むラベルを名称候補から除外します",
                    key='frm_name_exclude_terms')

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
        })
        st.session_state['saved_region_cfg'] = dict(region_cfg)
        st.session_state['region_cfg_is_saved'] = True
        st.toast("設定を保存しました", icon="✅")

    if st.session_state.get('region_cfg_is_saved'):
        st.caption("✅ 詳細設定：保存済み")

    # ============================================================
    # 領域を検出
    # ============================================================
    if st.button("領域を検出", key="detect_regions_btn"):
        try:
            with st.spinner('図面枠と矩形領域を検出中...'):
                analyses = {}
                for uf in uploaded_files:
                    tmp = save_uploadedfile(uf)
                    try:
                        analyses[uf.name] = analyze_dxf_regions(tmp, region_cfg)
                    finally:
                        try:
                            os.unlink(tmp)
                        except OSError:
                            pass
                st.session_state['region_analyses'] = analyses
                for k in list(st.session_state.keys()):
                    if isinstance(k, str) and k.startswith('rc_'):
                        del st.session_state[k]
                for k in ['excel_result', 'is_region_mode', 'region_results_summary']:
                    if k in st.session_state:
                        del st.session_state[k]
        except Exception as e:
            handle_error(e)

    # ============================================================
    # 領域の確認
    # ============================================================
    if 'region_analyses' in st.session_state:
        analyses = st.session_state['region_analyses']
        st.subheader("領域の確認")

        for fname, analysis in analyses.items():
            st.markdown(f"**{fname}**")
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
                    with st.popover("📐"):
                        st.markdown(f"**頂点の座標**（左下から / {len(corners)}点）")
                        st.code(
                            '\n'.join(
                                f"{i + 1}: ({x:.2f}, {y:.2f})"
                                for i, (x, y) in enumerate(corners)
                            ) or '(なし)'
                        )

                cands = reg['name_candidates']
                if not cands:
                    st.caption("　　（名称候補なし）")
                    continue

                # 他の図面/領域で選択済みの名称を収集し、一致する候補をデフォルトにする
                # ただし候補が1件のみ（選択肢なし・自動確定）の領域は除外する。
                # 候補1件の領域は「ユーザーが能動的に選んだわけではない確定」のため、
                # 同じラベルが隣接領域の候補にも上がる場合に誤って引き継がれるのを防ぐ。
                selected_elsewhere = set()
                for fn2, an2 in analyses.items():
                    for r2 in an2.get('regions', []):
                        if fn2 == fname and r2['id'] == reg['id']:
                            continue
                        cands2 = r2.get('name_candidates', [])
                        if len(cands2) <= 1:
                            continue  # 選択肢なし（自動確定）の領域はスキップ
                        for j, (_d2, t2) in enumerate(cands2):
                            if st.session_state.get(f"rc_{fn2}_{r2['id']}_{j}", False):
                                selected_elsewhere.add(t2)
                default_idx = 0
                for i, (d, t) in enumerate(cands):
                    if t in selected_elsewhere:
                        default_idx = i
                        break

                for i, (d, t) in enumerate(cands):
                    ck = f"rc_{fname}_{reg['id']}_{i}"
                    if ck not in st.session_state:
                        st.session_state[ck] = (i == default_idx)
                    st.checkbox(
                        f"　{t}　（境界線からの距離 {d:.0f}）",
                        key=ck,
                        on_change=_on_change_radio,
                        args=(fname, reg['id'], i, len(cands))
                    )

    # ============================================================
    # ラベルを抽出
    # ============================================================
    if st.button("ラベルを抽出"):
        try:
            with st.spinner(f'{len(uploaded_files)}個のDXFファイルを処理中...'):
                if 'region_analyses' in st.session_state:
                    # 領域付きモード
                    analyses = st.session_state['region_analyses']
                    name_selections = {}
                    for fname, analysis in analyses.items():
                        for reg in analysis.get('regions', []):
                            chosen = []
                            for i, (_d, t) in enumerate(reg['name_candidates']):
                                if st.session_state.get(f"rc_{fname}_{reg['id']}_{i}"):
                                    chosen.append(t)
                            if chosen:
                                name_selections[(fname, reg['id'])] = chosen

                    region_results = build_region_results(
                        analyses, name_selections, sort_value,
                        filter_circuit_only=filter_non_parts)

                    st.session_state.excel_result = create_region_excel_output(region_results)
                    st.session_state.output_filename = output_filename
                    st.session_state['is_region_mode'] = True
                    st.session_state['region_results_summary'] = {
                        f: {k: v for k, v in d.items() if k not in ('named', 'rows')}
                        for f, d in region_results.items()
                    }
                    st.session_state.processing_settings = {
                        'filter_option': filter_non_parts,
                        'extract_drawing_numbers': extract_drawing_numbers_option,
                        'extract_title': extract_title_option,
                    }

                else:
                    # 通常モード
                    temp_files = []
                    original_filenames = []
                    for uploaded_file in uploaded_files:
                        temp_file = save_uploadedfile(uploaded_file)
                        temp_files.append(temp_file)
                        original_filenames.append(uploaded_file.name)

                    results_temp = process_multiple_dxf_files(
                        temp_files,
                        filter_non_parts=filter_non_parts,
                        sort_order=sort_value,
                        debug=False,
                        selected_layers=selected_layers,
                        validate_ref_designators=validate_ref_designators,
                        extract_drawing_numbers_option=extract_drawing_numbers_option,
                        extract_title_option=extract_title_option,
                        original_filenames=original_filenames
                    )

                    results = {}
                    for temp_file, original_name in zip(temp_files, original_filenames):
                        if temp_file in results_temp:
                            labels, info = results_temp[temp_file]
                            info['filename'] = original_name
                            results[original_name] = (labels, info)

                    st.session_state.excel_result = create_excel_output(
                        results, filter_non_parts, sort_value, validate_ref_designators)
                    st.session_state.output_filename = output_filename
                    st.session_state['is_region_mode'] = False
                    st.session_state.results = results
                    st.session_state.processing_settings = {
                        'filter_option': filter_non_parts,
                        'validate_ref_designators': validate_ref_designators,
                        'sort_order': sort_value,
                        'extract_drawing_numbers': extract_drawing_numbers_option,
                        'extract_title': extract_title_option
                    }

                    for temp_file in temp_files:
                        try:
                            os.unlink(temp_file)
                        except Exception:
                            pass

        except Exception as e:
            handle_error(e)

    # ============================================================
    # 抽出結果
    # ============================================================
    if st.session_state.get('excel_result'):
        is_region_mode = st.session_state.get('is_region_mode', False)

        if is_region_mode:
            summ = st.session_state.get('region_results_summary', {})
            st.success(f"{len(summ)}個のDXFファイルからラベル抽出が完了しました（領域付き）")
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
            else:
                settings = st.session_state.get('processing_settings', {})
                results = st.session_state.get('results', {})
                rows = []
                for file_path, (labels, info) in results.items():
                    filename = info.get('filename', os.path.basename(file_path))
                    row = {
                        'ファイル名': filename,
                        '総抽出ラベル数': info.get('total_extracted', 0),
                        '除外ラベル数': info.get('filtered_count', 0),
                        '最終ラベル数': info.get('final_count', 0),
                        'レイヤー数': (
                            f"{info.get('processed_layers', 0)}"
                            f"/{info.get('total_layers', 0)}"
                        ),
                    }
                    if settings.get('extract_drawing_numbers'):
                        row['図番'] = info.get('main_drawing_number', '')
                        row['流用元図番'] = info.get('source_drawing_number', '')
                    if settings.get('extract_title'):
                        row['タイトル'] = info.get('title', '')
                        row['サブタイトル'] = info.get('subtitle', '')
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        st.write(f"出力ファイル：**{st.session_state.output_filename}**")
        st.download_button(
            label="Excelをダウンロード",
            data=st.session_state.excel_result,
            file_name=st.session_state.output_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch',
        )

        if st.button("🔄 新しい抽出を開始", key="restart_button"):
            for key in ['excel_result', 'output_filename', 'processing_settings',
                        'results', 'is_region_mode', 'region_analyses',
                        'region_results_summary']:
                if key in st.session_state:
                    del st.session_state[key]
            for k in list(st.session_state.keys()):
                if isinstance(k, str) and k.startswith('rc_'):
                    del st.session_state[k]
            st.rerun()


if __name__ == "__main__":
    app()
