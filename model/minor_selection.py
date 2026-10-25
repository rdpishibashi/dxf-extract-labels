"""小領域（`analysis['minor_regions']`）の採用と、「領域の確認」での上書き（streamlit 非依存・DXF-extract-labels 専用）。

小領域は、採用済みの領域名（`ref_designator.adopted_region_names()`）と一致する名称を持つものだけが採用される
（`ref_designator.add_minor_region_selections()` の規則）。「ラベルを抽出」と「領域の確認」の viewer が
この規則を同じ関数で使うので、画面と出力は食い違わない。

  - `adopted_minor_names()`: 採用される小領域 `{ファイルキー: {小領域id: 採用名}}`
  - `minor_name_selections()`: 採用された小領域に、viewer での「（領域選択しない）」「領域名を入力」を重ねた名称選択
"""
from . import ref_designator
from .region_choices import resolve_name_selections
from .region_decisions import DECIDED, RegionDecision


def adopted_minor_names(analyses, name_selections_by_file):
    """採用される小領域を `{ファイルキー: {小領域id: 採用名}}` で返す。
    採用の対象は、`name_selections_by_file`（通常の領域の名称選択）で採用された名称すべて。
    採用名は、候補が複数の対象名に一致するときは `OR_DELIMITER` で連結した名前のまま。"""
    targets = ref_designator.adopted_region_names(name_selections_by_file)
    merged = ref_designator.add_minor_region_selections(analyses, name_selections_by_file, targets)
    out = {}
    for fname, sel in merged.items():
        added = {rid: names[0] for (_f, rid), names in sel.items()
                 if (_f, rid) not in name_selections_by_file.get(fname, {}) and names}
        if added:
            out[fname] = added
    return out


def minor_name_selections(state, fname, analysis, minor_names):
    """この図面の採用小領域の名称選択 `{(ファイルキー, 小領域id): 名称のリスト}`。
    `minor_names` は `adopted_minor_names()` のこの図面の分（`{小領域id: 採用名}`）。

    「領域の確認」の上書き（`region_choices.resolve_name_selections()` と同じ規則）を重ねる:
    「（領域選択しない）」→ `['']`、入力名が図面枠内のラベルと一致 → その名称、不一致 → `['']`、
    上書きなし → 採用名。"""
    regions = [r for r in analysis.get('minor_regions', []) if r['id'] in minor_names]
    pseudo = {'regions': regions, 'labels': analysis.get('labels', [])}
    decisions = {r['id']: RegionDecision(DECIDED, name=minor_names[r['id']]) for r in regions}
    return resolve_name_selections(state, fname, pseudo, decisions)
