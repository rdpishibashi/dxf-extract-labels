"""Excel 出力モジュール

通常モード（ラベル抽出）・領域付きモード（矩形領域付きラベル抽出）の
Excel ファイル生成を担う。集計ロジック（build_region_results）は
region_detector モジュールに実装されている。

出力ファイルに記録するラベル・機器符号は `normalize_width()` で半角へ統一する
（領域付きモードの正規化は build_region_results 側で集計前に行われる）。
"""
import os
import pandas as pd
from io import BytesIO
from collections import Counter, defaultdict

from .common_utils import normalize_width
from .region_detector import build_region_label_summary


def create_excel_output(results, filter_non_parts, sort_option, validate_ref_designators):
    """通常モードの Excel を生成する。

    ラベルは半角へ正規化してから集計するため、図面上の表記が半角/全角どちら
    でも同じ語は同じ行（同じ個数）にまとまる。
    """
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

        summary_data = []
        invalid_by_symbol = defaultdict(lambda: {'count': 0, 'files': []})
        total_counter = Counter()

        for file_path, (labels, info) in results.items():
            filename = info.get('filename', os.path.basename(file_path))
            summary_data.append({
                'ファイル名': filename,
                '総抽出ラベル数': info.get('total_extracted', 0),
                '除外ラベル数': info.get('filtered_count', 0),
                '最終ラベル数': info.get('final_count', 0),
                '処理レイヤー数': info.get('processed_layers', 0),
                '全レイヤー数': info.get('total_layers', 0),
                '図番': info.get('main_drawing_number', ''),
                '流用元図番': info.get('source_drawing_number', ''),
                'タイトル': info.get('title', ''),
                'サブタイトル': info.get('subtitle', '')
            })

            total_counter.update(Counter(normalize_width(l) for l in labels))

            if validate_ref_designators and info.get('invalid_ref_designators'):
                for symbol in info['invalid_ref_designators']:
                    symbol = normalize_width(symbol)
                    invalid_by_symbol[symbol]['count'] += 1
                    if filename not in invalid_by_symbol[symbol]['files']:
                        invalid_by_symbol[symbol]['files'].append(filename)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        summary_worksheet = writer.sheets['Summary']
        for col_num, value in enumerate(summary_df.columns.values):
            summary_worksheet.write(0, col_num, value, header_format)

        for row_num, (file_path, (labels, info)) in enumerate(results.items(), start=1):
            filename = info.get('filename', os.path.basename(file_path))
            sheet_name = os.path.splitext(filename)[0][:31]
            summary_worksheet.write_url(
                row_num, 0, f"internal:'{sheet_name}'!A1", link_format, filename)

        total_data = [
            {'ラベル': lbl, '個数': total_counter[lbl]}
            for lbl in sorted(total_counter.keys())
        ]
        if total_data:
            total_df = pd.DataFrame(total_data)
            total_df.to_excel(writer, sheet_name='Total', index=False)
            total_worksheet = writer.sheets['Total']
            total_worksheet.write(0, 0, 'ラベル', header_format)
            total_worksheet.write(0, 1, '個数', header_format)
            total_worksheet.set_column('A:A', 25)
            total_worksheet.set_column('B:B', 10)

        for file_path, (labels, info) in results.items():
            filename = info.get('filename', os.path.basename(file_path))
            sheet_name = os.path.splitext(filename)[0][:31]

            counter = Counter(normalize_width(l) for l in labels)
            label_data = [
                {'ラベル': lbl, '個数': counter[lbl]}
                for lbl in sorted(counter.keys())
            ]

            if label_data:
                labels_df = pd.DataFrame(label_data)
                labels_df.to_excel(writer, sheet_name=sheet_name, index=False)
                worksheet = writer.sheets[sheet_name]
                worksheet.write(0, 0, 'ラベル', header_format)
                worksheet.write(0, 1, '個数', header_format)
                worksheet.set_column('A:A', 25)
                worksheet.set_column('B:B', 10)

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


def create_region_excel_output(region_results):
    """矩形領域抽出結果の Excel を生成する。各ファイルシートに『領域』列を付与する。"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        header_format = workbook.add_format({'bold': True, 'bg_color': '#D3D3D3', 'border': 1})
        link_format = workbook.add_format({'color': 'blue', 'underline': 1})

        summary_data = []
        for fname, data in region_results.items():
            summary_data.append({
                'ファイル名': fname,
                '総抽出ラベル数': data.get('total_in_frame', 0),
                '除外ラベル数': data.get('filtered_count', 0),
                '最終ラベル数': data.get('final_count', 0),
                '図面枠数': data['frames'],
                '検出領域数': data['regions_detected'],
                '確定領域数': data['regions_named'],
                '領域内ラベル数': data['in_region_count'],
            })
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        summary_ws = writer.sheets['Summary']
        for col_num, value in enumerate(summary_df.columns.values):
            summary_ws.write(0, col_num, value, header_format)
        for row_num, (fname, _data) in enumerate(region_results.items(), start=1):
            sheet_name = os.path.splitext(fname)[0][:31]
            summary_ws.write_url(row_num, 0, f"internal:'{sheet_name}'!A1", link_format, fname)
        summary_ws.set_column('A:A', 28)

        reg_rows = []
        for fname, data in region_results.items():
            for r in data['named']:
                reg_rows.append({
                    'ファイル名': fname,
                    'ページ No.': r['frame'] + 1,
                    '領域名': r['name'],
                    '面積率[%]': round(r['area_pct'], 1),
                    '領域内ラベル数': r['label_count'],
                })
        if reg_rows:
            reg_df = pd.DataFrame(reg_rows)
            reg_df.to_excel(writer, sheet_name='領域一覧', index=False)
            reg_ws = writer.sheets['領域一覧']
            for col_num, value in enumerate(reg_df.columns.values):
                reg_ws.write(0, col_num, value, header_format)
            reg_ws.set_column('A:A', 28)
            reg_ws.set_column('C:C', 22)

        # 領域別ラベル一覧: 領域名ごとに全ファイル横断でラベルを集計する
        # （領域名は同名なら複数ファイルにまたがって合算。詳細は
        # build_region_label_summary の docstring 参照）。
        files, label_summary_rows = build_region_label_summary(region_results)
        if label_summary_rows:
            lbl_ws = workbook.add_worksheet('領域別ラベル一覧')
            lbl_ws.write(0, 0, '領域名', header_format)
            lbl_ws.write(0, 1, 'ラベル', header_format)
            lbl_ws.write(0, 2, '合計個数', header_format)
            col = 3
            for _fname, _ident in files:
                lbl_ws.write(0, col, '図番', header_format)
                lbl_ws.write(0, col + 1, '個数', header_format)
                col += 2

            for row_num, r in enumerate(label_summary_rows, start=1):
                lbl_ws.write(row_num, 0, r['領域名'])
                lbl_ws.write(row_num, 1, r['ラベル'])
                lbl_ws.write(row_num, 2, r['合計個数'])
                col = 3
                for fname, ident in files:
                    lbl_ws.write(row_num, col, ident)
                    lbl_ws.write(row_num, col + 1, r['per_file'].get(fname, 0))
                    col += 2

            lbl_ws.set_column('A:A', 28)
            lbl_ws.set_column('B:B', 25)
            lbl_ws.set_column(2, 2, 10)
            lbl_ws.set_column(3, col - 1, 16)

        for fname, data in region_results.items():
            sheet_name = os.path.splitext(fname)[0][:31]
            df = pd.DataFrame(data['rows'], columns=['ラベル', '個数', '領域'])
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            for col_num, value in enumerate(['ラベル', '個数', '領域']):
                ws.write(0, col_num, value, header_format)
            ws.set_column('A:A', 25)
            ws.set_column('B:B', 8)
            ws.set_column('C:C', 40)

    output.seek(0)
    return output.getvalue()
