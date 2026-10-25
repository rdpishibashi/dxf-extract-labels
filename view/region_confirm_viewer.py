"""「領域の確認」の図面 viewer 版（streamlit 依存のビュー層。DXF-extract-labels 専用で、共有しない）。

全ファイルをファイル名のABC順の開閉枠に出し、枠の中の「図面を表示して領域を確認」を ON にすると
図面上の viewer（`st.components.v2` の双方向部品。選択を Streamlit へ返す）で領域の確認・選択・
名称の入力を行う。ページ上に選択肢は出さない（📐・行き止まり枝は枠内のポップオーバー）。
DXF-label-comparison・HostPL-comparer は、ページ上のチェックボックスで選ぶ共有の
`view/region_selection.render_region_confirmation_section()` を使う。

構成:
  - 選択状態の読み書きと遷移: `model/region_choices.py`（共有）
  - 領域の分類: `model/region_decisions.py`（共有）
  - 図面の SVG と viewer 用データ: `model/region_preview.py`（streamlit 非依存）
  - viewer 部品の HTML/CSS/JS: `view/assets/region_confirm_viewer.{html,css,js}`
    （同梱漏れでアプリ全体が起動不能にならないよう、初めて使うときに読む）
"""
from functools import lru_cache
from pathlib import Path

import streamlit as st

from model import minor_selection
from model import region_choices as rc
from model import region_decisions as rd
from model import region_preview
from model.ref_designator import unnamed_region_labels
from model.region_detector import pending_candidates_for_region
from view import region_selection as rs

REGION_CONFIRM_HEIGHT_PX = 600
ASSET_DIR = Path(__file__).resolve().parent / 'assets'
_ASSET_FILES = {
    'CONFIRM_HTML': 'region_confirm_viewer.html',
    'CONFIRM_CSS': 'region_confirm_viewer.css',
    'CONFIRM_JS': 'region_confirm_viewer.js',
}


@lru_cache(maxsize=None)
def _read_asset(name):
    return (ASSET_DIR / _ASSET_FILES[name]).read_text(encoding='utf-8')


def __getattr__(name):
    """`CONFIRM_HTML`/`CONFIRM_CSS`/`CONFIRM_JS`（viewer 部品の定義）を、使われたときに読む。"""
    if name in _ASSET_FILES:
        return _read_asset(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@st.cache_data(show_spinner=False, max_entries=20)
def render_region_base(dxf_bytes: bytes, version: int = region_preview.BASE_RENDER_VERSION):
    """元の図面全体の白 SVG（DXFバイト列と `BaseRender` の版をキーにキャッシュ）。
    `BaseRender` の項目を変えたら版を上げる（古い形のキャッシュが残って AttributeError に
    なるのを防ぐ。実行中のアプリはコードを更新してもキャッシュが残るため）。"""
    return region_preview.render_base_svg(dxf_bytes)


# 「領域の確認」の双方向 viewer（選択を Streamlit へ返す）。app.py は再実行のたびに読み込まれ、
# 同名の再登録は警告になるので、部品の定義（HTML/CSS/JS）が変わったときだけ登録し直す。
@st.cache_resource(show_spinner=False)
def _get_region_confirm_viewer(definition_sig):
    return st.components.v2.component(
        "region_confirm_viewer",
        html=_read_asset('CONFIRM_HTML'),
        css=_read_asset('CONFIRM_CSS'),
        js=_read_asset('CONFIRM_JS'),
    )


def mount_viewer(**kwargs):
    """viewer 部品を描く。テストは、この関数を差し替えて viewer の戻り値（choice）を模擬する。"""
    sig = hash((_read_asset('CONFIRM_HTML'), _read_asset('CONFIRM_CSS'), _read_asset('CONFIRM_JS')))
    return _get_region_confirm_viewer(sig)(**kwargs)


# ---------------------------------------------------------------------------
# 表示対象・領域の分類
# ---------------------------------------------------------------------------

def resolve(analyses):
    """全ファイルの最終結果 `{ファイルキー: minor_selection.Resolution}`（「ラベルを抽出」と同じ `resolve_all()`）。
    他のファイルでの選択・入力で変わるので、表示のたびに計算し直す（状態は書き換えない）。"""
    return minor_selection.resolve_all(st.session_state, analyses, rs.classify_file_regions)


def confirmation_targets(analyses, checked_names, resolutions=None):
    """「領域の確認」に表示するファイルを返す: 図面枠が検出できた全ファイル（図面枠未検出の既知の
    未対応ケースは除く。エラーのあるファイルは、エラーを見せるため含める）。
    戻り値はファイルキーでソートした `[(fname, analysis, pending_regions), ...]`。
    `pending_regions` は、候補が2件以上残り、別の図面枠でも決まらない（要選択の）領域。
    `resolutions`（`resolve()` の結果）を渡すと、採用された小領域が持つ名称で確定した領域（R1）を除いた要選択になる。"""
    excluded_names = rs.global_excluded_region_names()
    targets = []
    for fname, analysis in sorted(analyses.items()):
        err = analysis.get('error')
        if err and rs.FRAME_NOT_FOUND_ERROR_MARKER in err:
            continue  # 図面枠が見つからない既知の未対応ケースは表示しない
        decisions = (resolutions[fname].decisions if resolutions is not None
                     else rd.classify_regions(analysis, checked_names, excluded_names))
        pending_regions = [
            r for r in analysis.get('regions', []) if decisions[r['id']].status == rd.PENDING]
        targets.append((fname, analysis, pending_regions))
    return targets


def _unnamed_active_ids(decisions):
    """「除外」でない名称なし領域（`no name N`）の id 集合。除外した領域は確認の対象にしない。"""
    return {rid for rid, d in decisions.items() if d.status == rd.UNNAMED}


def needs_choice(analysis, pending_regions, checked_names, decisions=None):
    """このファイルに、ユーザーが確認する領域（要選択の領域、または除外でない名称なし領域）があるか。"""
    if decisions is None:
        decisions = rd.classify_regions(analysis, checked_names, rs.global_excluded_region_names())
    return bool(pending_regions or _unnamed_active_ids(decisions))


def _regions_with_status(analysis, decisions, *statuses):
    by_id = {r['id']: r for r in analysis.get('regions', [])}
    return [(by_id[rid], d) for rid, d in decisions.items()
            if d.status in statuses and rid in by_id]


def unnamed_regions(analysis, decisions=None):
    """名称の無い領域 `[(reg, 表示名, 特定ON?), ...]`（「抽出領域の選択」の `no name N` と同じ表示名）。
    「除外」した領域は含めない。"""
    decisions = decisions or rs.classify_file_regions(analysis)
    return [(reg, d.name, d.adopted)
            for reg, d in _regions_with_status(analysis, decisions, rd.UNNAMED)]


def decided_regions(analysis, pending_regions=None, decisions=None):
    """要選択ではないが名称が確定している領域 `[(reg, 名称), ...]`（viewer の赤表示用）。"""
    decisions = decisions or rs.classify_file_regions(analysis)
    return [(reg, d.name) for reg, d in _regions_with_status(
        analysis, decisions, rd.DECIDED, rd.OTHER_FRAME)]


def minor_regions_for_viewer(analyses, fname):
    """この図面の採用される小領域 `[(reg, 採用名), ...]`（viewer に確定領域として出す。`resolve()` の結果）。"""
    return resolve(analyses)[fname].minors


def inactive_regions(analysis, decisions=None):
    """抽出されない領域 `[(reg, 表示名, 注記), ...]`（「除外」、または名称候補を1つも特定していない
    領域。viewer では灰色で表示する）。"""
    decisions = decisions or rs.classify_file_regions(analysis)
    labels = unnamed_region_labels(analysis)
    out = []
    for reg, d in _regions_with_status(analysis, decisions, rd.EXCLUDED, rd.UNSPECIFIED):
        cands = reg.get('name_candidates', [])
        name = cands[0][1] if cands else labels.get(reg['id'], "（名称なし）")
        out.append((reg, name, "除外" if d.status == rd.EXCLUDED else "特定OFF"))
    return out


def apply_viewer_choice(fname, reg, kind, checked_names, idx=None, text='', typed_text=None):
    """viewer（ブラウザ内）での選択を反映する（遷移の本体は `model/region_choices.apply_viewer_choice()`）。"""
    rc.apply_viewer_choice(
        st.session_state, fname, reg, kind, checked_names, idx=idx, text=text, typed_text=typed_text)


# ---------------------------------------------------------------------------
# 描画
# ---------------------------------------------------------------------------

def render_info_rows(regions):
    """選択の必要がある領域ごとに、📐（頂点の座標）と⚠️（行き止まり枝）をポップオーバーで出す。
    開閉枠の中に置くため、エクスパンダーは使えない（入れ子不可）のでポップオーバーにする。"""
    for reg in regions:
        col_header, col_c, col_d = st.columns([6, 1, 2])
        col_header.markdown(f"　**図面{reg['frame'] + 1} / 領域{reg['id'] + 1}**　面積 {reg['area_pct']:.0f}%")
        with col_c:
            rs.render_corners_popover(reg.get('corners', []))
        dangling = reg.get('dangling_edges', [])
        if dangling:
            with col_d, st.popover(f"⚠️ 行き止まり枝（{len(dangling)} 件）"):
                st.caption(
                    "この領域の境界探索から除外された、どこにも閉じていない線分（境界線と同じ線種 "
                    "lineweight=25/color=2）です。手描きの作画ミスの可能性があるため、該当する handle "
                    "を確認してください。")
                lines = []
                for br in dangling:
                    att = br.get('attachment')
                    lines.append(f"取り付け点 {f'({att[0]:.2f}, {att[1]:.2f})' if att else '(不明)'}:")
                    for ent in br['entities']:
                        (sx, sy), (ex, ey) = ent['start'], ent['end']
                        lines.append(f"  handle {ent['handle'] or '(handle不明)'}: "
                                     f"({sx:.2f}, {sy:.2f}) - ({ex:.2f}, {ey:.2f})")
                st.code('\n'.join(lines))


def show_viewer(fname, dxf_bytes, analysis, pending_regions, locked, analyses=None, resolution=None):
    """「領域の確認」の viewer。viewer 内の選択を選択状態（`rc_…`）へ反映する。
    採用される小領域も、確定した領域として出す。`resolution`（`resolve()` のこのファイルの結果）が無ければ計算する。"""
    checked = rs.global_checked_region_names()
    if resolution is None:
        resolution = resolve(analyses or {fname: analysis})[fname]
    decisions = resolution.decisions
    minors = resolution.minors
    decided = decided_regions(analysis, decisions=decisions) + minors
    unnamed = unnamed_regions(analysis, decisions)
    # 確定済み・無名・小領域も「領域名を入力」で名称を変更できる（状態を初期化）
    for reg, *_rest in decided + unnamed:
        rc.ensure_typed_defaults(st.session_state, fname, reg['id'])
    try:
        base = render_region_base(dxf_bytes)
        data = region_preview.build_confirmation_data(
            base, fname, analysis, pending_regions, decided,
            st.session_state, checked, locked,
            rs.match_typed_region_name, pending_candidates_for_region,
            unnamed=unnamed, inactive=inactive_regions(analysis, decisions),
            minor_ids={r['id'] for r, _name in minors}, hidden_cands=resolution.hidden)
        data['height'] = REGION_CONFIRM_HEIGHT_PX  # viewer の外枠の高さ（JS が明示的に設定する）
        res = mount_viewer(
            key=f"regprev_viewer_{fname}", data=data, height=REGION_CONFIRM_HEIGHT_PX,
            on_choice_change=lambda: None)
    except Exception as e:
        st.warning(f"プレビューを表示できませんでした: {e}")
        return
    choice = res.choice if hasattr(res, "choice") else None
    seq_key = f"regprev_seq_{fname}"
    if choice and choice.get('seq') != st.session_state.get(seq_key) and not locked:
        st.session_state[seq_key] = choice.get('seq')
        reg = next((r for r in list(analysis.get('regions', [])) + list(analysis.get('minor_regions', []))
                    if r['id'] == choice.get('region')), None)
        if reg is not None:
            apply_viewer_choice(
                fname, reg, choice.get('kind'), checked, idx=choice.get('idx'),
                text=choice.get('text', ''), typed_text=choice.get('typed_text'))
            st.rerun()


def _render_file(fname, analysis, pending_regions, analyses, checked_names, locked, files, resolution):
    """「領域の確認」の1ファイル分（開閉枠の中）。"""
    if rs.render_confirmation_file_header(fname, analysis, pending_regions):
        return  # エラー（警告を出した）
    if not analysis.get('regions'):
        st.caption("領域は検出されませんでした。")
        return

    # 選択状態を表示順に初期化する。viewer ON/OFF のどちらでも先に行う
    rs.ensure_pending_defaults(fname, pending_regions, analyses, checked_names)

    uf = files.get(fname)
    if uf is None:  # アップロードが残っていないファイルは図面を出せない（ページ上の選択肢で選ぶ）
        rs.render_pending_region_blocks(fname, pending_regions, checked_names, locked)
        return

    def _only_this_viewer(fname_):
        """図面の描画は重いので、同時に開く viewer は1ファイルだけにする。"""
        for other in files:
            if other != fname_:
                st.session_state[f"regprev_confirm_{other}"] = False

    if not st.toggle("図面を表示して領域を確認", value=False, key=f"regprev_confirm_{fname}",
                     on_change=_only_this_viewer, args=(fname,)):
        # OFF: 選択・入力はすべて図面（viewer）の中で行う。ここでは状況だけを示す
        decisions = resolution.decisions
        editable_ids = [rid for rid, d in decisions.items()
                        if d.status in (rd.DECIDED, rd.OTHER_FRAME, rd.UNNAMED)]
        minors = resolution.minors
        editable_ids += [r['id'] for r, _name in minors]
        n_changed = sum(
            1 for rid in editable_ids
            if rc.get_value(st.session_state, fname, rid, 'typed')
            or rc.get_value(st.session_state, fname, rid, 'skip'))
        minor_note = f"・小領域 {len(minors)} 個" if minors else ""
        st.caption(
            f"検出領域 {len(analysis.get('regions', []))} 個"
            f"（要選択 {len(pending_regions)} 個{minor_note}・図面で名称を変更または選択しない {n_changed} 個）。"
            "選択・名称の入力は、図面を表示して行います。")
    else:
        show_viewer(fname, uf.getvalue(), analysis, pending_regions, locked, analyses, resolution)  # ON: 選択は viewer の中で行う

    # 開閉枠の中に、選択が必要な領域の📐・行き止まり枝を出す
    unnamed_ids = _unnamed_active_ids(resolution.decisions)
    render_info_rows([
        r for r in analysis.get('regions', [])
        if r in pending_regions or r['id'] in unnamed_ids])


def render_confirmation_section(analyses, locked, uploaded_files=()):
    """「領域の確認」セクション（DXF-extract-labels 版）。図面枠が検出できた全ファイルを、ファイル名の
    ABC順の開閉枠（見出しに「（エラー）」「（選択の必要あり）」「（確定のみ）」）で表示し、各枠の中で
    図面上の viewer を開いて領域の確認・選択・名称の入力を行う（同時に開くのは1ファイル）。
    表示するかどうか（領域検出済みか）は呼び出し側が判定する。"""
    files = {uf.name: uf for uf in uploaded_files}
    checked_names = rs.global_checked_region_names()
    targets = confirmation_targets(analyses, checked_names)
    if not targets:
        return

    # 小領域の採用と要選択の領域の確定は全ファイルの選択から決まるので、描画の前に、全ファイルの
    # 既定の選択（共有の分類で要選択の領域）を初期化してから、最終結果を1回だけ計算する
    for fname, analysis, pending_regions in targets:
        if analysis.get('regions'):
            rs.ensure_pending_defaults(fname, pending_regions, analyses, checked_names)
    resolutions = resolve(analyses)
    targets = confirmation_targets(analyses, checked_names, resolutions)

    st.subheader("領域の確認")
    st.caption(
        "各ファイルを開き、図面で領域の確認・選択・名称の入力を行います。確定した領域の名称も、図面の「領域名を入力」で変更できます。"
        "検出した領域の名称はすべて抽出の対象です（抽出しない領域は「（領域選択しない）」にします）。"
        "面積が小さい領域は、他のファイルで採用された名称と一致するときだけ採用され、図面に「小領域」として表示します。")

    def _needs_choice(item):
        _f, a, pend = item
        return needs_choice(a, pend, checked_names, resolutions[_f].decisions)

    n_choice = sum(1 for it in targets if _needs_choice(it))
    if n_choice:
        st.caption(f"選択の必要あり: {n_choice} ファイル")
        if st.toggle("選択の必要ありのファイルのみ表示", value=False, key="regconfirm_only_action"):
            targets = [it for it in targets if _needs_choice(it)]

    for item in targets:
        fname, analysis, pending_regions = item
        suffix = "（エラー）" if analysis.get('error') else (
            "（選択の必要あり）" if _needs_choice(item) else "（確定のみ）")
        container = st.expander(
            f"{fname}{suffix}", expanded=bool(st.session_state.get(f"regprev_confirm_{fname}")))
        with container:
            _render_file(fname, analysis, pending_regions, analyses, checked_names, locked, files,
                         resolutions[fname])
