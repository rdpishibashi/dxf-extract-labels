"""Reference Designator 抽出検討ツール

extracted_labels*.xlsx（DXF-extract-labels の「機器符号（候補）以外も抽出」ON
で出力した Excel。`Total` シートに ラベル・個数 列を持つ）を複数入力し、
reference_designator_candidates.xlsx と同じ構成の分析用 Excel を生成する。

パターン・除外リストの定義は `utils/ref_designator.py` を単一の正として再利用
する（本体アプリの判定ロジックと乖離しないため、ここで独自に定義し直さない）。

起動方法:
    streamlit run tools/reference_designator_analyzer.py
"""
import glob
import os
import re
import sys
from collections import Counter, defaultdict
from io import BytesIO

import openpyxl
import streamlit as st

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from utils import ref_designator as rd  # noqa: E402

st.set_page_config(
    page_title="Reference Designator 抽出検討ツール",
    page_icon="🔍",
    layout="wide",
)


# ============================================================
# 入力ファイルの収集・読み込み
# ============================================================

def _iter_input_sources(uploaded_files, folder_paths, glob_pattern, recursive):
    """(表示名, バイト列 or ローカルパス) のリストを返す。"""
    sources = []
    for uf in uploaded_files or []:
        sources.append((uf.name, uf.getvalue()))

    for folder in folder_paths:
        folder = folder.strip()
        if not folder:
            continue
        if not os.path.isdir(folder):
            st.warning(f"フォルダが見つかりません: {folder}")
            continue
        pattern_path = os.path.join(folder, '**', glob_pattern) if recursive \
            else os.path.join(folder, glob_pattern)
        found = glob.glob(pattern_path, recursive=recursive)
        for path in sorted(found):
            base = os.path.basename(path)
            if base.startswith('~$'):
                continue
            sources.append((base, path))
    return sources


def _read_total_sheet(name, source):
    """(表示名, バイト列またはパス) から Total シートの (ラベル, 個数) を読む。"""
    try:
        target = BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
        wb = openpyxl.load_workbook(target, read_only=True, data_only=True)
    except Exception as e:
        st.warning(f"{name}: 読み込みに失敗しました（{e}）")
        return []
    if 'Total' not in wb.sheetnames:
        st.warning(f"{name}: Total シートが見つかりません（スキップ）")
        wb.close()
        return []
    ws = wb['Total']
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[0] is None:
            continue
        label = str(row[0])
        count = int(row[1]) if row[1] is not None else 0
        rows.append((label, count))
    wb.close()
    return rows


def aggregate_labels(sources):
    """各ソースの Total シートを正規化して集計する。

    戻り値: (agg, per_source_stats)
      agg: {正規化ラベル: {'count': int, 'files': set(表示名)}}
      per_source_stats: [{'ファイル名': str, 'ラベル種類数': int}]
    """
    agg = defaultdict(lambda: {'count': 0, 'files': set()})
    per_source_stats = []
    for name, source in sources:
        rows = _read_total_sheet(name, source)
        per_file = Counter()
        for label, count in rows:
            norm = rd.normalize_label(label)
            if not norm:
                continue
            per_file[norm] += count
        for norm, count in per_file.items():
            agg[norm]['count'] += count
            agg[norm]['files'].add(name)
        per_source_stats.append({'ファイル名': name, 'ラベル種類数': len(per_file)})
    return agg, per_source_stats


# ============================================================
# パターン表記（電気技術者向け人間可読ノーテーション）
# ============================================================
#
# 「数字列の前の部分はそのまま」「直後の数字は桁数表記(1,12,123...)」「その後ろは
# +英字繰返し=A*/数字繰返し=1*/ハイフンはそのまま、後ろに何も続かなければ+1*に
# 集約」という表記ルール（2026-07 ユーザーと確定）。本体アプリでは使わない
# 分析専用の表記のため、ref_designator.py には持ち込まずここに閉じる。

_ASCENDING_DIGITS = '123456789'
_TAIL_TOKEN_RE = re.compile(r'[A-Z]+|[0-9]+|-')
_PREFIX_DIGITS_TAIL_HYPHEN_RE = re.compile(r'^([A-Z]+-[A-Z]+)([0-9]+)([A-Z0-9-]*)$')
_PREFIX_DIGITS_TAIL_RE = re.compile(r'^([A-Z]+)([0-9]+)([A-Z0-9-]*)$')


def _ascending_digit_repr(n: int) -> str:
    if n <= len(_ASCENDING_DIGITS):
        return _ASCENDING_DIGITS[:n]
    return ''.join(str((i % 9) + 1) for i in range(n))


def _encode_tail(tail: str) -> str:
    out = []
    for tok in _TAIL_TOKEN_RE.findall(tail):
        if tok == '-':
            out.append('-')
        elif tok.isdigit():
            out.append('1*')
        else:
            out.append('A*')
    return ''.join(out)


def build_pattern_signature(label: str, pattern_name: str):
    """ラベルとその一致パターン名から人間可読のパターン表記を作る。"""
    if pattern_name == 'letters_only':
        return label
    rx = _PREFIX_DIGITS_TAIL_HYPHEN_RE if pattern_name == 'hyphen_letters_digits_any' \
        else _PREFIX_DIGITS_TAIL_RE
    m = rx.match(label)
    if not m:
        return None
    prefix, digits, tail = m.groups()
    if tail == '':
        return f'{prefix}+1*'
    return f'{prefix}{_ascending_digit_repr(len(digits))}+{_encode_tail(tail)}'


# ============================================================
# 分類・Excel 生成
# ============================================================

def classify_aggregated_labels(agg):
    """agg（aggregate_labels の戻り値）を分類する。

    戻り値: (ref_designator_rows, signature_rows, exclusion_impact)
      ref_designator_rows: [{'ラベル','個数','出現ファイル数','一致パターン',
                              '除外カテゴリ','除外ステータス'}]（Patterns一致のみ）
      signature_rows: [{'パターン表記','個数','出現ファイル数','該当ラベル数'}]
      exclusion_impact: {カテゴリ名: {'labels': int, 'count': int}}
    """
    ref_rows = []
    sig_agg = defaultdict(lambda: {'count': 0, 'files': set(), 'labels': set()})
    exclusion_impact = defaultdict(lambda: {'labels': 0, 'count': 0})

    for label, data in agg.items():
        judgment = rd._judgment_text(label)
        status, category = rd.classify_judgment_detailed(judgment)
        if status == 'no_match':
            continue
        pattern_name = rd.matched_pattern_name(judgment)
        ref_rows.append({
            'ラベル': label,
            '個数': data['count'],
            '出現ファイル数': len(data['files']),
            '一致パターン': pattern_name,
            '除外カテゴリ': category or '',
            '除外ステータス': '確定' if status == 'excluded' else '',
        })
        if status == 'excluded':
            exclusion_impact[category]['labels'] += 1
            exclusion_impact[category]['count'] += data['count']

        sig = build_pattern_signature(judgment, pattern_name)
        if sig:
            sig_agg[sig]['count'] += data['count']
            sig_agg[sig]['files'] |= data['files']
            sig_agg[sig]['labels'].add(label)

    ref_rows.sort(key=lambda r: r['ラベル'])
    signature_rows = [
        {'パターン表記': sig, '個数': d['count'], '出現ファイル数': len(d['files']),
         '該当ラベル数': len(d['labels'])}
        for sig, d in sig_agg.items()
    ]
    signature_rows.sort(key=lambda r: r['パターン表記'])
    return ref_rows, signature_rows, exclusion_impact


def build_output_workbook(ref_rows, signature_rows, exclusion_impact):
    """reference_designator_candidates.xlsx と同じ構成の Excel を作る。"""
    wb = openpyxl.Workbook()

    ws1 = wb.active
    ws1.title = 'ReferenceDesignators'
    ws1.append(['ラベル', '個数', '出現ファイル数', '一致パターン', '除外カテゴリ', '除外ステータス'])
    for r in ref_rows:
        ws1.append([r['ラベル'], r['個数'], r['出現ファイル数'], r['一致パターン'],
                    r['除外カテゴリ'], r['除外ステータス']])
    for col, width in zip('ABCDEF', (20, 10, 14, 26, 26, 12)):
        ws1.column_dimensions[col].width = width
    ws1.freeze_panes = 'A2'

    ws2 = wb.create_sheet('Patterns')
    ws2.append(['パターン名', '正規表現', '説明'])
    for name, rx, desc in rd.PATTERN_CATEGORIES:
        ws2.append([name, rx.pattern, desc])
    ws2.append(['combined（統合）', rd.CANDIDATE_PATTERN.pattern,
                '上記3パターンのいずれかに一致（判定前に NFKC 正規化・前後空白除去・括弧以降除去を適用）'])
    for col, width in zip('ABC', (24, 60, 60)):
        ws2.column_dimensions[col].width = width

    ws3 = wb.create_sheet('PatternSignatures')
    ws3.append(['パターン表記', '個数', '出現ファイル数', '該当ラベル数'])
    for r in signature_rows:
        ws3.append([r['パターン表記'], r['個数'], r['出現ファイル数'], r['該当ラベル数']])
    for col, width in zip('ABCD', (28, 10, 14, 12)):
        ws3.column_dimensions[col].width = width
    ws3.freeze_panes = 'A2'

    ws4 = wb.create_sheet('ExclusionPatterns')
    ws4.append(['カテゴリ', 'ステータス', '種別', 'パターン/一覧', '理由', '該当ラベル数', '該当個数合計'])
    for name, (words, desc) in rd.EXCLUSION_EXACT_CATEGORIES.items():
        imp = exclusion_impact.get(name, {'labels': 0, 'count': 0})
        ws4.append([name, '確定', '完全一致（列挙）', ', '.join(sorted(words)), desc,
                    imp['labels'], imp['count']])
    for name, rx, desc in rd.EXCLUSION_REGEX_CATEGORIES:
        imp = exclusion_impact.get(name, {'labels': 0, 'count': 0})
        ws4.append([name, '確定', '正規表現', rx.pattern, desc, imp['labels'], imp['count']])
    for col, width in zip('ABCDEFG', (26, 10, 22, 60, 45, 12, 14)):
        ws4.column_dimensions[col].width = width
    ws4.freeze_panes = 'A2'

    ws5 = wb.create_sheet('RemainingUnclassified')
    ws5.append(['ラベル', '個数', '出現ファイル数', '一致パターン'])
    remaining = [r for r in ref_rows if r['除外ステータス'] == '']
    remaining.sort(key=lambda r: -r['個数'])
    for r in remaining:
        ws5.append([r['ラベル'], r['個数'], r['出現ファイル数'], r['一致パターン']])
    for col, width in zip('ABCD', (20, 10, 14, 20)):
        ws5.column_dimensions[col].width = width
    ws5.freeze_panes = 'A2'

    output = BytesIO()
    wb.save(output)
    output.seek(0)
    return output.getvalue()


# ============================================================
# Streamlit UI
# ============================================================

def app():
    st.title("Reference Designator 抽出検討ツール")
    st.write(
        "複数の `extracted_labels*.xlsx`（Total シート）からラベルを集計し、"
        "機器符号（候補）パターン・除外パターンの判定結果を "
        "`reference_designator_candidates.xlsx` と同じ構成の Excel にまとめます。"
    )
    with st.expander("ℹ️ このツールについて", expanded=False):
        st.info(
            "DXF-extract-labels 本体の「機器符号（候補）以外も抽出」ON で出力した "
            "extracted_labels*.xlsx を入力とし、Reference Designator の抽出パターン・"
            "除外パターンを検討するための分析用ツールです。\n\n"
            "パターン・除外リストの定義は `utils/ref_designator.py`（本体アプリの判定"
            "ロジック）を単一の正として参照するため、本体の挙動とここでの分析結果は"
            "常に一致します。\n\n"
            "**本体アプリとの違い**: 本体アプリは DXF ファイルを直接解析し、図面枠・"
            "図面情報欄（フォーマットブロック）を構造的に判定して除外します。本ツールは"
            "集計済みラベル一覧（Total シート、座標情報なし）だけを見るため、その構造的"
            "除外は行えません。人名等は個別リストを持たない設計のため、本ツールの結果には"
            "図面情報欄由来のラベル（人名等）がパターン一致・除外非該当のまま残ることが"
            "あります（本体アプリの実際の出力ではこれらは構造的除外で消えます）。"
        )

    st.subheader("入力")
    col_left, col_right = st.columns(2)
    with col_left:
        uploaded_files = st.file_uploader(
            "extracted_labels*.xlsx をアップロード（複数可）",
            type="xlsx",
            accept_multiple_files=True,
        )
    with col_right:
        folder_text = st.text_area(
            "フォルダパス（1行に1つ、ローカルのフルパス）",
            value="",
            help="このマシン上のディレクトリを指定すると、配下の xlsx ファイルも"
                 "対象に加えます。アップロードと併用できます。",
        )
        glob_pattern = st.text_input("検索パターン（glob）", value="extracted_labels*.xlsx")
        recursive = st.checkbox("サブフォルダも検索する", value=True)

    folder_paths = [line for line in folder_text.splitlines() if line.strip()]

    if st.button("候補を抽出", type="primary"):
        sources = _iter_input_sources(uploaded_files, folder_paths, glob_pattern, recursive)
        if not sources:
            st.warning("入力ファイルがありません。アップロードするか、フォルダを指定してください。")
        else:
            with st.spinner(f"{len(sources)} 個のファイルを処理中..."):
                agg, per_source_stats = aggregate_labels(sources)
                ref_rows, signature_rows, exclusion_impact = classify_aggregated_labels(agg)
                excel_bytes = build_output_workbook(ref_rows, signature_rows, exclusion_impact)

            st.session_state['rda_result'] = {
                'excel_bytes': excel_bytes,
                'sources': [name for name, _ in sources],
                'per_source_stats': per_source_stats,
                'total_labels': len(agg),
                'pattern_matched': len(ref_rows),
                'excluded': sum(1 for r in ref_rows if r['除外ステータス']),
                'remaining': sum(1 for r in ref_rows if not r['除外ステータス']),
                'exclusion_impact': dict(exclusion_impact),
                'signature_count': len(signature_rows),
            }
            st.rerun()

    result = st.session_state.get('rda_result')
    if result:
        st.subheader("結果")
        st.success(f"{len(result['sources'])} 個のファイルを処理しました。")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("正規化後ラベル種類数", result['total_labels'])
        m2.metric("Patterns 一致", result['pattern_matched'])
        m3.metric("除外", result['excluded'])
        m4.metric("残存（機器符号候補）", result['remaining'])

        with st.expander("📊 入力ファイル別の内訳", expanded=False):
            import pandas as pd
            st.dataframe(pd.DataFrame(result['per_source_stats']), width='stretch', hide_index=True)

        with st.expander("📊 除外カテゴリ別の内訳", expanded=False):
            import pandas as pd
            rows = [
                {'カテゴリ': name, '該当ラベル数': imp['labels'], '該当個数合計': imp['count']}
                for name, imp in sorted(result['exclusion_impact'].items(),
                                         key=lambda kv: -kv[1]['labels'])
            ]
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        st.download_button(
            label="Excelをダウンロード（reference_designator_candidates.xlsx 形式）",
            data=result['excel_bytes'],
            file_name="reference_designator_candidates.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )

        if st.button("🔄 リセット"):
            del st.session_state['rda_result']
            st.rerun()


if __name__ == "__main__":
    app()
