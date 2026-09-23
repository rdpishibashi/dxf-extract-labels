"""「結線図-組立図を比較」オプション（v3.6.0）のドメインロジック。

DXF-label-compare プロジェクトの「結線図-組立図比較」機能（Excel入力）を、
DXFから抽出したラベルデータ（メモリ上の抽出結果）を直接比較する形に移植したもの。
Excel読み込みを経由しない点、機器符号候補列を追加した点が移植元との違い
（詳細は Tools/CLAUDE.md 参照）。streamlit に依存しない純関数のみで構成する。

比較対象は以下の2グループ（両方ともタイトルによる自動判定）:
  - A（結線図）: タイトルに「UNIT内結線図」を含む全ファイルの合算
  - B（組立図）: タイトルに「組立図」を含むファイル。ちょうど1件のときだけ
    自動確定し、0件・複数件の場合は呼び出し側（app.py）が候補から選ばせる

`results` は `ref_designator.build_ref_designator_final()`（通常モード）または
`build_ref_designator_region_results()`（領域モード）が返す辞書
（{ファイル名: {...}}）をそのまま受け取る。両モードで図番のキー名が異なる
（通常モード='main_drawing_number'、領域モード='drawing_number'）ため、
`drawing_ident()` で吸収する。

比較キーは「"("・"（"（半角/全角とも）以降を除去した文字列」を使う
（2026-09-14 ユーザー指示、`_compare_key()`）。括弧内の補足（抵抗値・
ワイヤ色コード等）だけが違う同一部品を、比較上は別ラベル扱いにしないため。
比較シートに表示する「ラベル」列もこの切り詰め後の文字列を使う。
"""
import os

from .common_utils import normalize_width
from .ref_designator import is_ref_designator_label

WIRING_TITLE_TEXT = 'UNIT内結線図'
ASSEMBLY_TITLE_TEXT = '組立図'

COMPARE_SHEET_NAME = '結線図-組立図比較'
COMPARE_COLUMNS = (
    '機器符号候補', 'ラベル', 'A 合計出現数', 'A 図番数', 'B 出現数', 'A 図番', '比較結果',
)

KUBUN_BOTH = '両方'
KUBUN_A_ONLY = 'A のみ'
KUBUN_B_ONLY = 'B のみ'

# 表示色（Excel出力で使用）。DXF-label-compare の配色をそのまま踏襲する
# （区分が「両方」でも個数が一致するかどうかで色を分けるため、区分そのものとは
# 別に「表示スタイル区分」（ROW_STYLE_*）を設ける）。
ROW_STYLE_A_ONLY = 'A_ONLY'
ROW_STYLE_B_ONLY = 'B_ONLY'
ROW_STYLE_MISMATCH = 'MISMATCH'
ROW_STYLE_MATCH = 'MATCH'

COLOR_A_ONLY = {'bg_color': '#D9E1F2', 'font_color': '#1F4E79'}      # 青
COLOR_B_ONLY = {'bg_color': '#C6EFCE', 'font_color': '#006100'}      # 緑
COLOR_MISMATCH = {'bg_color': '#FFEB9C', 'font_color': '#9C6500'}    # 黄

# ROW_STYLE_* -> 色（MATCH は無色のため None）
ROW_STYLE_COLORS = {
    ROW_STYLE_A_ONLY: COLOR_A_ONLY,
    ROW_STYLE_B_ONLY: COLOR_B_ONLY,
    ROW_STYLE_MISMATCH: COLOR_MISMATCH,
    ROW_STYLE_MATCH: None,
}


def row_style(kubun: str, a_count: int, b_count: int) -> str:
    """区分と個数から表示スタイル区分（ROW_STYLE_*）を返す。

    - 区分が『A のみ』『B のみ』ならそのまま対応する区分を返す（青／緑）。
    - 区分が『両方』の場合、A個数とB個数が一致すれば無色（MATCH）、
      不一致なら黄（MISMATCH）。
    """
    if kubun == KUBUN_A_ONLY:
        return ROW_STYLE_A_ONLY
    if kubun == KUBUN_B_ONLY:
        return ROW_STYLE_B_ONLY
    if a_count == b_count:
        return ROW_STYLE_MATCH
    return ROW_STYLE_MISMATCH


def _title_contains(title, text) -> bool:
    """タイトルを NFKC 正規化・大文字化してから text を含むか判定する。

    実データ検証（2026-09-13）で「UNIT内結線図」表記が全角・半角混在
    （全角『ＵＮＩＴ内結線図』13件・半角『UNIT内結線図』3件、計16件中）
    だったため、正規化を省くと大半を取りこぼす。`normalize_width()` は
    かな・漢字には影響しない（`model/common_utils.py` 参照）。
    """
    if not title:
        return False
    return text in normalize_width(title).upper()


def _compare_key(label: str) -> str:
    """比較用のラベル文字列を返す（"("・"（"〈半角/全角とも〉以降を除去し、
    直前の空白があれば `rstrip()` で併せて除去する）。

    `ref_designator_patterns._judgment_text()` の括弧処理と同じ規約に
    倣うが、あちらは半角の `(` のみを対象にするのに対し、こちらは
    全角の `（` も対象にする（2026-09-14 ユーザー指示）。括弧内の補足
    （抵抗値・ワイヤ色コード等、例 `R10(2.2K)`・`FBWH（白）`）だけが
    異なる同一部品を、A・B比較上は同一ラベルとして扱うため。
    括弧を含まない場合は元の文字列をそのまま返す。
    """
    idx = min((i for i in (label.find('('), label.find('（')) if i >= 0), default=-1)
    return label[:idx].rstrip() if idx >= 0 else label


def drawing_ident(fname: str, data: dict) -> str:
    """ファイルの図番識別子を返す（通常モード/領域モードのキー名差を吸収）。

    通常モードは 'main_drawing_number'、領域モードは 'drawing_number' に
    図番が入る（`ref_designator.build_ref_designator_final`/
    `build_ref_designator_region_results` 参照）。どちらも未抽出の場合は
    ファイル名（拡張子なし）にフォールバックする
    （`excel_output._build_total_rows` 系と同じ方式）。
    """
    return (data.get('main_drawing_number') or data.get('drawing_number')
            or os.path.splitext(fname)[0])


def select_wiring_files(results: dict) -> list:
    """`results` からタイトルに「UNIT内結線図」を含む全ファイル名を返す（A側）。"""
    return sorted(
        fname for fname, data in results.items()
        if _title_contains(data.get('title'), WIRING_TITLE_TEXT)
    )


def select_assembly_candidates(results: dict) -> list:
    """`results` からタイトルに「組立図」を含む全ファイル名を返す（B側候補）。

    0件・1件・複数件のいずれもありうる。`judge_drawing_types()` の初期判定
    （組立図欄のチェック初期値）に使う。最終的にどのファイルをB（組立図）と
    するかは「図面種別判定結果」セクション（app.py）でユーザーが確認・
    修正した選択（`validate_type_selection()` の入力）で決まる。
    """
    return sorted(
        fname for fname, data in results.items()
        if _title_contains(data.get('title'), ASSEMBLY_TITLE_TEXT)
    )


# ============================================================
# 図面種別判定結果（「結線図-組立図を比較」オプション、v3.7.0）
# ============================================================
#
# 抽出結果からA（結線図）・B（組立図）をタイトルで自動判定し、「図面種別判定
# 結果」セクション（app.py）でユーザーが一覧・修正できるようにする。以下の
# 3関数はいずれもstreamlitに依存しない純関数——app.py側はチェックボックスの
# 状態（selection）をこれらの関数に渡すだけで、判定・比較可否・表示行の
# いずれも計算できる。

ERROR_NO_WIRING = '結線図（タイトルに「UNIT内結線図」を含む図面）が選択されていません。'
ERROR_NO_ASSEMBLY = '組立図が選択されていません。'
ERROR_MULTIPLE_ASSEMBLY_TEMPLATE = '組立図が{n}件選択されています。1件だけ選択してください。'
ERROR_OVERLAP_TEMPLATE = '同じ図面が結線図・組立図の両方に選択されています: {files}'


def judge_drawing_types(results: dict) -> dict:
    """`results` の全ファイルについて、タイトルから結線図・組立図の初期判定
    （チェックボックスの初期値）を行う。

    戻り値: `{fname: {'wiring': bool, 'assembly': bool}}`（`results` の
    全ファイルを含む。両方Trueの初期判定は起こらないが——同一ファイルの
    タイトルが「UNIT内結線図」「組立図」を両方含む場合は起こりうるため、
    その場合は`validate_type_selection()`側で不成立として扱う）。
    """
    wiring = set(select_wiring_files(results))
    assembly = set(select_assembly_candidates(results))
    return {
        fname: {'wiring': fname in wiring, 'assembly': fname in assembly}
        for fname in results
    }


def validate_type_selection(selection: dict) -> dict:
    """判定結果（`{fname: {'wiring': bool, 'assembly': bool}}`）から、
    「結線図-組立図比較」を実行できるかどうかを判定する。

    成立条件: 結線図が1件以上・組立図がちょうど1件・同一ファイルへの
    重複チェックなし。いずれかを満たさない場合は `errors` に該当理由を
    すべて列挙する（複数該当しうる。例: 結線図0件かつ組立図0件）。

    戻り値: `{'wiring_files': [...], 'b_fname': str|None, 'errors': [...]}`
      `errors` が空なら比較可能（`b_fname` は確定した組立図ファイル名）。
      1件以上あれば比較不可（`b_fname` は常に None）。
    """
    wiring_files = sorted(f for f, v in selection.items() if v.get('wiring'))
    assembly_files = sorted(f for f, v in selection.items() if v.get('assembly'))
    overlap = sorted(set(wiring_files) & set(assembly_files))

    errors = []
    if not wiring_files:
        errors.append(ERROR_NO_WIRING)
    if not assembly_files:
        errors.append(ERROR_NO_ASSEMBLY)
    elif len(assembly_files) > 1:
        errors.append(ERROR_MULTIPLE_ASSEMBLY_TEMPLATE.format(n=len(assembly_files)))
    if overlap:
        errors.append(ERROR_OVERLAP_TEMPLATE.format(files='、'.join(overlap)))

    b_fname = assembly_files[0] if (not errors and len(assembly_files) == 1) else None
    return {'wiring_files': wiring_files, 'b_fname': b_fname, 'errors': errors}


def type_judgment_rows(results: dict, selection: dict) -> list:
    """「図面種別判定結果」表示用の行データをファイル名昇順で返す。

    各行: `{'ファイル名': str, '結線図': bool, '組立図': bool, '図番': str,
    'タイトル': str, 'サブタイトル': str}`。`ファイル名` は表の列としては
    表示しない（呼び出し側でDataFrameのindexに使う）想定だが、行とファイル名
    の対応付けに必要なためここでは含める。図番は `drawing_ident()`
    （未抽出時はファイル名〈拡張子なし〉にフォールバック）。
    """
    rows = []
    for fname in sorted(results.keys()):
        data = results[fname]
        sel = selection.get(fname, {})
        rows.append({
            'ファイル名': fname,
            '結線図': bool(sel.get('wiring')),
            '組立図': bool(sel.get('assembly')),
            '図番': drawing_ident(fname, data),
            'タイトル': data.get('title') or '',
            'サブタイトル': data.get('subtitle') or '',
        })
    return rows


def compare_assembly(results: dict, wiring_files: list, b_fname: str) -> dict:
    """A（`wiring_files` の合算）とB（`b_fname` 単体）のラベルを比較する。

    `results[*]['rows']` は `[{'機器符号候補': 'Y'|None, 'ラベル': str, '個数': int, ...}]`
    （通常モード・領域モードとも共通のキーを持つ。領域モードの '領域' キーは無視する）。
    集計・比較・表示のいずれも `row['ラベル']` を `_compare_key()` で切り詰めた
    文字列をキーとして使う（括弧以降の補足違いを同一部品として合算するため）。

    戻り値: {'b_file': str, 'rows': [...], 'summary': {...}}
      rows は `COMPARE_COLUMNS` の7列をラベル昇順で持つ。
      summary は画面表示・Excel出力のいずれにも使わず、テスト検証用。
    """
    a_labels: dict = {}
    for fname in wiring_files:
        data = results[fname]
        ident = drawing_ident(fname, data)
        for row in data.get('rows', []):
            label = _compare_key(row['ラベル'])
            entry = a_labels.setdefault(label, {'count': 0, 'idents': set()})
            entry['count'] += int(row['個数'])
            entry['idents'].add(ident)

    b_data = results[b_fname]
    b_labels: dict = {}
    for row in b_data.get('rows', []):
        label = _compare_key(row['ラベル'])
        b_labels[label] = b_labels.get(label, 0) + int(row['個数'])

    labels = sorted(set(a_labels) | set(b_labels))
    rows = []
    for label in labels:
        in_a, in_b = label in a_labels, label in b_labels
        kubun = KUBUN_BOTH if in_a and in_b else (KUBUN_A_ONLY if in_a else KUBUN_B_ONLY)
        a_entry = a_labels.get(label)
        a_count = a_entry['count'] if a_entry else 0
        b_count = b_labels.get(label, 0)
        rows.append({
            '機器符号候補': 'Y' if is_ref_designator_label(label) else None,
            'ラベル': label,
            'A 合計出現数': a_count,
            'A 図番数': len(a_entry['idents']) if a_entry else 0,
            'B 出現数': b_count,
            'A 図番': ', '.join(sorted(a_entry['idents'])) if a_entry else '',
            '比較結果': kubun,
        })

    a_idents = {ident for entry in a_labels.values() for ident in entry['idents']}
    summary = {
        'A 対象図番数': len(a_idents),
        'A ユニークラベル数': len(a_labels),
        'B ユニークラベル数': len(b_labels),
        '両方': sum(1 for r in rows if r['比較結果'] == KUBUN_BOTH),
        'A のみ': sum(1 for r in rows if r['比較結果'] == KUBUN_A_ONLY),
        'B のみ': sum(1 for r in rows if r['比較結果'] == KUBUN_B_ONLY),
    }

    return {'b_file': b_fname, 'rows': rows, 'summary': summary}
