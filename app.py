import streamlit as st
import os
import tempfile
import sys
from pathlib import Path
import pandas as pd
from io import BytesIO

# utils モジュールをインポート可能にするためのパスの追加
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, 'utils')
sys.path.insert(0, utils_path)

from utils.extract_labels import extract_labels, get_layers_from_dxf, process_multiple_dxf_files
from utils.common_utils import save_uploadedfile, handle_error

st.set_page_config(
    page_title="DXF Extract Labels",
    page_icon="📝",
    layout="wide",
)

def create_excel_output(results, filter_option, sort_option, validate_ref_designators):
    from collections import Counter, defaultdict

    output = BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book

        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D3D3D3',
            'border': 1
        })
        link_format = workbook.add_format({
            'color': 'blue',
            'underline': 1
        })

        # サマリーシートの作成
        summary_data = []
        invalid_by_symbol = defaultdict(lambda: {'count': 0, 'files': []})
        total_counter = Counter()

        for file_path, (labels, info) in results.items():
            filename = info.get('filename', os.path.basename(file_path))
            summary_data.append({
                'ファイル名': filename,
                '総抽出数': info.get('total_extracted', 0),
                'フィルタリング除外数': info.get('filtered_count', 0),
                '最終ラベル数': info.get('final_count', 0),
                '処理レイヤー数': info.get('processed_layers', 0),
                '全レイヤー数': info.get('total_layers', 0),
                '図番': info.get('main_drawing_number', ''),
                '流用元図番': info.get('source_drawing_number', ''),
                'タイトル': info.get('title', ''),
                'サブタイトル': info.get('subtitle', '')
            })

            total_counter.update(Counter(labels))

            if validate_ref_designators and info.get('invalid_ref_designators'):
                for symbol in info['invalid_ref_designators']:
                    invalid_by_symbol[symbol]['count'] += 1
                    if filename not in invalid_by_symbol[symbol]['files']:
                        invalid_by_symbol[symbol]['files'].append(filename)

        # Summary シートの書き込み
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        summary_worksheet = writer.sheets['Summary']
        for col_num, value in enumerate(summary_df.columns.values):
            summary_worksheet.write(0, col_num, value, header_format)

        # ファイル名セルに各ファイルシートへの内部リンクを設定
        for row_num, (file_path, (labels, info)) in enumerate(results.items(), start=1):
            filename = info.get('filename', os.path.basename(file_path))
            sheet_name = os.path.splitext(filename)[0][:31]
            summary_worksheet.write_url(row_num, 0, f"internal:'{sheet_name}'!A1", link_format, filename)

        # Total シートの作成（Summary の直後）
        total_data = [{'ラベル': lbl, '個数': total_counter[lbl]} for lbl in sorted(total_counter.keys())]
        if total_data:
            total_df = pd.DataFrame(total_data)
            total_df.to_excel(writer, sheet_name='Total', index=False)

            total_worksheet = writer.sheets['Total']
            total_worksheet.write(0, 0, 'ラベル', header_format)
            total_worksheet.write(0, 1, '個数', header_format)
            total_worksheet.set_column('A:A', 25)
            total_worksheet.set_column('B:B', 10)

        # 各ファイルの詳細シートを作成
        for file_path, (labels, info) in results.items():
            filename = info.get('filename', os.path.basename(file_path))
            sheet_name = os.path.splitext(filename)[0][:31]

            counter = Counter(labels)
            label_data = [{'ラベル': lbl, '個数': counter[lbl]} for lbl in sorted(counter.keys())]

            if label_data:
                labels_df = pd.DataFrame(label_data)
                labels_df.to_excel(writer, sheet_name=sheet_name, index=False)

                worksheet = writer.sheets[sheet_name]
                worksheet.write(0, 0, 'ラベル', header_format)
                worksheet.write(0, 1, '個数', header_format)
                worksheet.set_column('A:A', 25)
                worksheet.set_column('B:B', 10)

        # Invalid シートの作成（該当データがある場合のみ）
        if invalid_by_symbol:
            invalid_data = [
                {
                    '機器符号': sym,
                    '個数': data['count'],
                    'ファイル名': ', '.join(data['files'])
                }
                for sym, data in sorted(invalid_by_symbol.items())
            ]
            invalid_df = pd.DataFrame(invalid_data)
            invalid_df.to_excel(writer, sheet_name='Invalid', index=False)

            invalid_worksheet = writer.sheets['Invalid']
            for col_num, value in enumerate(invalid_df.columns.values):
                invalid_worksheet.write(0, col_num, value, header_format)
            invalid_worksheet.set_column('A:A', 20)
            invalid_worksheet.set_column('B:B', 8)
            invalid_worksheet.set_column('C:C', 60)

    output.seek(0)
    return output.getvalue()

def app():
    st.title('DXF Extract Labels')
    st.write('DXFファイルからテキストラベルを抽出し、Excel形式で出力します。')

    # プログラム説明
    with st.expander("ℹ️ プログラム説明", expanded=False):
        help_text = [
            "このツールは、DXFファイルからテキスト要素（ラベル）を抽出し、Excelファイルに出力します。",
            "",
            "**使用手順：**",
            "1. DXFファイルをアップロードしてください（複数可）",
            "2. レイヤー選択（必要な場合のみ）",
            "3. 必要に応じてオプション設定を調整します",
            "4. 「ラベルを抽出」ボタンをクリックして処理を実行します",
            "",
            "**Excelファイルの内容：**",
            "- サマリーシート：全ファイルの抽出結果概要",
            "- 各ファイルシート：個別ファイルの抽出ラベル一覧",
            "- Invalidシート（妥当性チェック有効時）：適合しない機器符号のリスト",
            "",
            "**高度な機能：**",
            "- 機器符号（回路記号）のみを抽出するフィルタリング",
            "- 機器符号の妥当性チェック（標準フォーマットとの適合性）",
            "- ラベルの並び替え（昇順、降順、並び替えなし）",
            "- レイヤー選択による抽出範囲の制限",
            "- 図面番号の自動抽出"
        ]

        st.info("\n".join(help_text))

    # ファイルアップロード
    st.subheader("DXFファイルのアップロード")
    uploaded_files = st.file_uploader(
        "DXFファイルを選択してください（複数可）",
        type="dxf",
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)}個のファイルが選択されました")

        # レイヤー選択機能
        st.subheader("レイヤー選択（オプション）")

        # 最初のファイルからレイヤー一覧を取得
        temp_file_path = save_uploadedfile(uploaded_files[0])
        available_layers = get_layers_from_dxf(temp_file_path)
        os.unlink(temp_file_path)

        if available_layers:
            layer_selection_enabled = st.checkbox(
                "特定のレイヤーのみを処理する",
                value=False,
                help="チェックを入れると、選択したレイヤーのみを処理対象とします"
            )

            selected_layers = None
            if layer_selection_enabled:
                selected_layers = st.multiselect(
                    "処理対象とするレイヤーを選択してください",
                    options=available_layers,
                    default=available_layers,
                    help="複数選択可能です。デフォルトでは全レイヤーが選択されています。"
                )

                if selected_layers:
                    st.info(f"{len(selected_layers)}個のレイヤーが選択されています")
        else:
            selected_layers = None

        # オプション設定
        with st.expander("オプション設定", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                filter_option = st.checkbox(
                    "機器符号（候補）のみ抽出",
                    value=False,
                    help="以下のパターンに一致するラベルのみを機器符号として抽出します："
                         "\n\n【基本パターン】"
                         "\n• 英文字のみ: CNCNT, FB"
                         "\n• 英文字+数字: R10, CN3, PSW1"
                         "\n• 英文字+数字+英文字: X14A, RMSS2A"
                         "\n\n【括弧付きパターン】"
                         "\n• 英文字(補足): FB(), MSS(MOTOR)"
                         "\n• 英文字+数字(補足): R10(2.2K), MSSA(+)"
                         "\n• 英文字+数字+英文字(補足): U23B(DAC)"
                         "\n\n※英文字だけの場合は英文字2個以上、それ以外の場合は英文字1個以上、数字1個以上必要です"
                )

                # 機器符号妥当性チェックオプション
                validate_ref_designators = False
                if filter_option:
                    validate_ref_designators = st.checkbox(
                        "機器符号妥当性チェック",
                        value=False,
                        help="抽出された機器符号がフォーマットに適合するかチェックします。"
                             "\n適合しない機器符号のリストを別シートに出力します。"
                             "\n（例：CBnnn, ELB(CB) nnn, R, Annn等の標準フォーマット）"
                    )

                # 図面番号抽出オプション
                extract_drawing_numbers_option = st.checkbox(
                    "図面番号を抽出",
                    value=False,
                    help="DXFファイルから図面番号（例：DE5313-008-02B）を抽出します。"
                         "\n抽出された図面番号はサマリーシートに表示されます。"
                )

                # タイトル抽出オプション
                extract_title_option = st.checkbox(
                    "タイトルとサブタイトルを抽出",
                    value=False,
                    help="DXFファイルからタイトルとサブタイトルを抽出します。"
                         "\n「TITLE」ラベルの右側近辺、「REVISION」の下方向に配置されたテキストから抽出します。"
                         "\n抽出されたタイトルとサブタイトルはサマリーシートに表示されます。"
                )

            with col2:
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

                # 出力ファイル名設定
                output_filename = st.text_input(
                    "出力Excelファイル名",
                    value="extracted_labels.xlsx",
                    help="出力するExcelファイルの名前を指定します"
                )
                if not output_filename.endswith('.xlsx'):
                    output_filename += '.xlsx'

        # 処理実行ボタン
        if st.button("ラベルを抽出"):
            try:
                with st.spinner(f'{len(uploaded_files)}個のDXFファイルを処理中...'):
                    # 一時ファイルに保存
                    temp_files = []
                    original_filenames = []
                    for uploaded_file in uploaded_files:
                        temp_file = save_uploadedfile(uploaded_file)
                        temp_files.append(temp_file)
                        original_filenames.append(uploaded_file.name)

                    # ラベル抽出
                    results_temp = process_multiple_dxf_files(
                        temp_files,
                        filter_non_parts=filter_option,
                        sort_order=sort_value,
                        debug=False,
                        selected_layers=selected_layers,
                        validate_ref_designators=validate_ref_designators,
                        extract_drawing_numbers_option=extract_drawing_numbers_option,
                        extract_title_option=extract_title_option,
                        original_filenames=original_filenames
                    )

                    # 結果のキーを一時ファイルパスから元のファイル名に置き換え
                    results = {}
                    for temp_file, original_name in zip(temp_files, original_filenames):
                        if temp_file in results_temp:
                            labels, info = results_temp[temp_file]
                            # 元のファイル名で情報を更新
                            info['filename'] = original_name
                            results[original_name] = (labels, info)

                    # Excel出力を生成
                    excel_data = create_excel_output(
                        results,
                        filter_option,
                        sort_value,
                        validate_ref_designators
                    )

                    # セッション状態に保存
                    st.session_state.excel_result = excel_data
                    st.session_state.output_filename = output_filename
                    st.session_state.processing_settings = {
                        'filter_option': filter_option,
                        'validate_ref_designators': validate_ref_designators,
                        'sort_order': sort_value,
                        'extract_drawing_numbers': extract_drawing_numbers_option,
                        'extract_title': extract_title_option
                    }
                    st.session_state.results = results

                    # 一時ファイルの削除
                    for temp_file in temp_files:
                        try:
                            os.unlink(temp_file)
                        except:
                            pass

            except Exception as e:
                handle_error(e)

        # セッション状態に保存された結果を表示
        if 'excel_result' in st.session_state and st.session_state.excel_result:
            settings = st.session_state.get('processing_settings', {})
            results = st.session_state.get('results', {})

            # 結果サマリーの表示
            st.success(f"{len(results)}個のDXFファイルからラベル抽出が完了しました")

            # 統計情報の表示
            with st.expander("📊 抽出結果統計", expanded=False):
                stats_rows = []
                for file_path, (labels, info) in results.items():
                    filename = info.get('filename', os.path.basename(file_path))
                    row = {
                        'ファイル名': filename,
                        '総抽出数': info.get('total_extracted', 0),
                        'フィルタリング除外数': info.get('filtered_count', 0),
                        '最終ラベル数': info.get('final_count', 0),
                        'レイヤー数': f"{info.get('processed_layers', 0)}/{info.get('total_layers', 0)}",
                    }
                    if settings.get('extract_drawing_numbers'):
                        row['図番'] = info.get('main_drawing_number', '')
                        row['流用元図番'] = info.get('source_drawing_number', '')
                    if settings.get('extract_title'):
                        row['タイトル'] = info.get('title', '')
                        row['サブタイトル'] = info.get('subtitle', '')
                    stats_rows.append(row)
                st.dataframe(pd.DataFrame(stats_rows), width='stretch', hide_index=True)

            # ダウンロードボタンの表示
            st.subheader("結果のダウンロード")
            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**出力ファイル**: {st.session_state.output_filename}")

            with col2:
                st.download_button(
                    label="Excelをダウンロード",
                    data=st.session_state.excel_result,
                    file_name=st.session_state.output_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            # 新しい抽出を開始するボタン
            if st.button("🔄 新しい抽出を開始", key="restart_button"):
                # セッション状態をクリアして新しい抽出を開始
                for key in ['excel_result', 'output_filename', 'processing_settings', 'results']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    else:
        st.info("DXFファイルをアップロードしてください")

if __name__ == "__main__":
    app()
