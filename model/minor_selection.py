"""小領域（`analysis['minor_regions']`）の採用と、「領域の確認」での上書き（streamlit 非依存・DXF-extract-labels 専用）。

小領域は、採用済みの領域名（`ref_designator.adopted_region_names()`）と一致する名称を持つものだけが採用される
（`ref_designator.add_minor_region_selections()` の規則）。「ラベルを抽出」と「領域の確認」の viewer が
この規則を同じ関数で使うので、画面と出力は食い違わない。

  - `adopted_minor_names()`: 採用される小領域 `{ファイルキー: {小領域id: 採用名}}`
  - `minor_name_selections()`: 採用された小領域に、viewer での「（領域選択しない）」「領域名を入力」を重ねた名称選択
  - `resolve_all()`: 全ファイルの最終結果（通常領域の分類・名称選択・採用小領域）を決まった順序で1回計算する。
    「領域の確認」の viewer と「ラベルを抽出」は、この関数だけを使う（画面と出力が食い違わない）

採用された小領域が持つ名称は、同じ図面枠の要選択の通常領域の選択肢から外す（R1）。残りが1件なら確定する。
小領域を「（領域選択しない）」にするか、入力名で別のラベルにすると、その名称は選択肢に戻る。
"""
from collections import defaultdict
from dataclasses import dataclass, field

from . import ref_designator
from . import region_choices as rc
from .common_utils import normalize_width
from .region_choices import resolve_name_selections
from .region_decisions import DECIDED, PENDING, RegionDecision

OR_DELIMITER = ref_designator.OR_DELIMITER


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


@dataclass
class Resolution:
    """1ファイルの最終結果（`resolve_all()` の戻り値の値）。
    decisions: 通常領域の分類（R1 適用後。`{領域id: RegionDecision}`）
    selections: 通常領域と採用小領域の最終の名称選択 `{(ファイルキー, 領域id): [名称]}`
    minors: viewer に出す採用小領域 `[(reg, 採用名), ...]`（「領域選択しない」「入力名」で上書きされていても含む）
    hidden: 選択肢から外した候補 `{要選択の領域id: {候補の文字列}}`"""
    decisions: dict
    selections: dict
    minors: list = field(default_factory=list)
    hidden: dict = field(default_factory=dict)


def resolve_all(state, analyses, classify):
    """全ファイルの最終結果 `{ファイルキー: Resolution}` を返す。状態（`state`）は書き換えない（純関数）。

    `classify(analysis)` は共有の分類（`region_decisions.classify_regions()` を「特定」「除外」の状態で呼ぶ
    呼び出し側の関数。アプリでは `region_selection.classify_file_regions`）。順序は固定（変えない。変えると
    小領域の採用と要選択の領域の確定が互いに依存して結果が定まらなくなる）:
      1. 共有の分類 d0 と、全ファイルの名称選択（要選択の領域の選択を含む）
      2. 採用対象の名称 = 手順1で採用された名称すべて
      3. 小領域の採用: 手順1の選択から**要選択の領域の分を除いて**採用規則を適用する
         （要選択の領域で今選ばれている候補は、同じ図面枠の小領域の採用を妨げない。ここで循環を断つ）
      4. 小領域に「（領域選択しない）」「領域名を入力」の上書きを重ねる
      5. 同じ図面枠の要選択の領域から、手順4の小領域が持つ名称を候補から外す（残り1件なら確定）
      6. 通常領域の名称選択を作り直し、同じ図面枠で通常領域が採用した名称を、上書きのない小領域の名称から外す
    手順6の後始末は、手順5には戻さない（連鎖させない）。"""
    d0 = {f: classify(a) for f, a in analyses.items()}
    sel_full = {f: resolve_name_selections(state, f, a, d0[f]) for f, a in analyses.items()}
    targets = ref_designator.adopted_region_names(sel_full)
    sel_fixed = {f: {k: v for k, v in sel.items() if d0[f][k[1]].status != PENDING}
                 for f, sel in sel_full.items()}
    merged = ref_designator.add_minor_region_selections(analyses, sel_fixed, targets)

    out = {}
    for f, analysis in analyses.items():
        adopted = {rid: names[0] for (_f, rid), names in merged.get(f, {}).items()
                   if (_f, rid) not in sel_fixed[f] and names}
        minor_sel = minor_name_selections(state, f, analysis, adopted)
        minor_by_id = {r['id']: r for r in analysis.get('minor_regions', [])}

        held = defaultdict(set)  # 図面枠 -> 小領域が持つ名称（NFKC）
        for (_f, rid), names in minor_sel.items():
            for n in names:
                for part in n.split(OR_DELIMITER):
                    if part:
                        held[minor_by_id[rid]['frame']].add(normalize_width(part))

        decisions = dict(d0[f])
        hidden = {}
        for reg in analysis.get('regions', []):
            d = d0[f][reg['id']]
            if d.status != PENDING:
                continue
            removed = [t for _d, t in d.candidates if normalize_width(t) in held[reg['frame']]]
            reduced = [(dist, t) for dist, t in d.candidates if t not in removed]
            if not removed or not reduced:
                continue  # 外すものが無い／全候補を小領域が持つ（そのまま選ばせる）
            hidden[reg['id']] = set(removed)
            decisions[reg['id']] = (
                RegionDecision(DECIDED, name=reduced[0][1]) if len(reduced) == 1
                else RegionDecision(PENDING, candidates=tuple(reduced)))

        selections = resolve_name_selections(state, f, analysis, decisions)
        frame_of = {r['id']: r['frame'] for r in analysis.get('regions', [])}
        taken = defaultdict(set)  # 図面枠 -> 通常領域が採用した名称（NFKC）
        for (_f, rid), names in selections.items():
            for n in names:
                if n:
                    taken[frame_of[rid]].add(normalize_width(n))

        minors = []
        for rid, name in adopted.items():
            reg = minor_by_id[rid]
            overridden = (rc.get_value(state, f, rid, 'skip') or rc.get_value(state, f, rid, 'typed'))
            if overridden:
                selections[(f, rid)] = minor_sel[(f, rid)]
                minors.append((reg, name))
                continue
            parts = [p for p in name.split(OR_DELIMITER) if normalize_width(p) not in taken[reg['frame']]]
            if parts:
                name = OR_DELIMITER.join(parts)
                selections[(f, rid)] = [name]
                minors.append((reg, name))
        out[f] = Resolution(decisions, selections, minors, hidden)
    return out
