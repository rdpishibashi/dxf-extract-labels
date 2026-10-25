"""抽出結果プレビューの「図面を表示」viewer のデータ（モデル層。streamlit 非依存・DXF-extract-labels 専用）。

「抽出結果」の各図面シートで、「機器符号候補」の選択（DEF／CAN／（空欄））に当たるラベルを、
図面上で元の文字の色を変えて示す。図面の描画と、ラベルごとの CSS クラスは
`model/region_preview.render_base_svg()`（`BaseRender.label_classes`）。領域の確認 viewer と同じ図面を使う。

ここで作るもの:
  - `kept_label_positions()`: 抽出で残ったラベルの位置（領域付きモードでは「除外」領域のラベルを除く）
  - `sheet_to_file()`: プレビューのシート名 -> ファイル名
  - `build_label_viewer_data()`: viewer 部品に渡す data（JSON 化できる dict）
"""
import os
from typing import Dict, List, Optional, Sequence, Tuple

from model import region_preview
from model.ref_designator import (
    MARK_CANDIDATE, MARK_DEFINED, excluded_region_ids, filter_labels_outside_excluded)

BLANK_MARK = '（空欄）'  # 「機器符号候補」列が空欄の区分（画面の絞り込み選択欄の値と同じ）
MARK_ORDER = (MARK_DEFINED, MARK_CANDIDATE, BLANK_MARK)
MARK_COLORS: Dict[str, str] = {
    MARK_DEFINED: "#ff3030",    # DEF
    MARK_CANDIDATE: "#ff9f1a",  # CAN
    BLANK_MARK: "#ffe600",      # 空欄
}
REGION_LINE_COLOR = "#ff3030"  # 抽出された領域の領域線（輪郭のみ。塗りつぶさない）
EXCEL_SHEET_NAME_MAX = 31


def kept_label_positions(labels: Sequence[Tuple[str, float, float]], analysis: Optional[dict] = None,
                         excluded_names=frozenset()) -> List[Tuple[str, float, float]]:
    """抽出結果に残るラベルの位置 `[(text, x, y), ...]` を返す。

    領域付きモード（`analysis` あり）では、「除外」指定された領域（境界線上を含む）のラベルを取り除く。
    `ref_designator.build_ref_designator_region_results()` が Excel の行を作る前に行うのと同じ処理
    （`excluded_region_ids()`・`filter_labels_outside_excluded()`）。"""
    labels = list(labels)
    if not analysis or not excluded_names:
        return labels
    excl_ids = excluded_region_ids(analysis, excluded_names)
    excl_polys = [reg['polygon'] for reg in analysis.get('regions', []) if reg['id'] in excl_ids]
    return filter_labels_outside_excluded(labels, excl_polys)


def sheet_name_of(fname: str) -> str:
    """ファイル名に対応する Excel のシート名（`model/excel_output.py` と同じ規則: 拡張子なし・31文字まで）。"""
    return os.path.splitext(fname)[0][:EXCEL_SHEET_NAME_MAX]


def sheet_to_file(sheet: str, fnames: Sequence[str]) -> Optional[str]:
    """プレビューのシート名に対応するファイル名を返す。Summary・Total・領域一覧など、
    ファイルに対応しないシートは None（31文字で切り詰めた名前が重なるときは先頭のファイル）。"""
    for fname in fnames:
        if sheet_name_of(fname) == sheet:
            return fname
    return None


def mark_of_label(rows: Sequence[dict]) -> Dict[str, str]:
    """抽出結果の行（`{'ラベル': ..., '機器符号候補': 'DEF'|'CAN'|None}`）から、ラベル -> 区分
    （`MARK_ORDER` のいずれか。空欄は `BLANK_MARK`）を作る。表と同じ値を使うため。"""
    out = {}
    for r in rows:
        mark = r.get('機器符号候補')
        out[r['ラベル']] = mark if mark in (MARK_DEFINED, MARK_CANDIDATE) else BLANK_MARK
    return out


def region_outlines(base: region_preview.BaseRender, named_regions: Sequence[dict]) -> List[dict]:
    """抽出された領域（`build_named_regions()` の named。Excel に出力される領域と同じ）の輪郭を、SVG 座標の
    `[{id, points}, ...]` にする。同じ領域に複数の名称があっても線は1本（領域 id で重複を除く）。"""
    seen = set()
    out = []
    for reg in named_regions:
        rid = reg.get('id')
        poly = reg.get('polygon') or []
        if rid in seen or len(poly) < 3:
            continue
        seen.add(rid)
        out.append({"id": rid, "points": [list(region_preview._to_svg(base.matrix, x, y)) for x, y in poly]})
    return out


def build_label_viewer_data(base: region_preview.BaseRender, fname: str,
                            positions: Sequence[Tuple[str, float, float]],
                            mark_of: Dict[str, str], selected_marks,
                            named_regions: Sequence[dict] = ()) -> dict:
    """viewer 部品に渡す data を作る。

    positions: `kept_label_positions()` の結果／mark_of: `mark_of_label()`／selected_marks: 色を付ける区分の集合
    groups: 区分ごとの `{mark, color, count（出現数。選択に関わらず全ラベル）, selected, placed（図面に描かれている数）}`
    items: 色を付けるラベルの CSS クラス `[{m: 区分, c: [[クラス名, 'fill'|'stroke'], ...]}, ...]`
    regions: 抽出された領域の輪郭 `[{id, points}]`（領域付きモードのみ。`region_outlines()`）
    unplaced: 選択した区分のうち、図面に描かれていない（CSS クラスが無い）ラベルの数。ハイライトできず、
      位置を推測して重ね書きもしない（Paper Space のみの文字などは図面の座標系が違うため）。"""
    selected = set(selected_marks)
    counts = {m: 0 for m in MARK_ORDER}
    placed = {m: 0 for m in MARK_ORDER}
    items = []
    unplaced = 0
    for text, x, y in positions:
        mark = mark_of.get(text, BLANK_MARK)
        if mark not in counts:
            mark = BLANK_MARK
        counts[mark] += 1
        classes = region_preview.label_classes_of(base, (text, x, y))
        if classes:
            placed[mark] += 1
        if mark not in selected:
            continue
        if classes:
            items.append({"m": mark, "c": classes})
        else:
            unplaced += 1
    return {
        "fname": fname, "viewBox": [base.view_box[0], base.view_box[1]],
        "svg": base.inner_svg, "sig": f"{fname}:{len(base.inner_svg)}:{hash(base.inner_svg) & 0xffffffff}",
        "empty": base.empty, "hasFont": base.cjk_font is not None,
        "colors": dict(MARK_COLORS), "regionColor": REGION_LINE_COLOR,
        "regions": region_outlines(base, named_regions),
        "groups": [{"mark": m, "color": MARK_COLORS[m], "count": counts[m], "placed": placed[m],
                    "selected": m in selected} for m in MARK_ORDER],
        "items": items, "unplaced": unplaced,
    }
