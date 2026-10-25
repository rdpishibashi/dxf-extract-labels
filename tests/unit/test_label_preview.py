"""
このテストが守るもの: 抽出結果プレビューの「図面を表示」（v3.17.0）のデータ作り（model/label_preview.py）と、
画面（app.py の `_render_preview_viewer`）の出し分け。
- 抽出で残ったラベルの位置は、領域付きモードでは「除外」領域のラベルを含まない（Excel と同じ）
- シート名 -> ファイル名は Excel と同じ規則（拡張子なし・31文字）。図面に対応しないシートでは出さない
- 色を付けるのは選択した区分だけ。図面に描かれていないラベルは件数にだけ出し、位置を推測して重ね書きしない
- 領域付きモードでは、抽出された領域の領域線（輪郭のみ・領域ごとに1本）を重ねる。通常モードでは出さない
"""
import base64
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ezdxf  # noqa: E402
import io  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from model import label_preview as lp  # noqa: E402
from model import region_preview as rp  # noqa: E402
from model.excel_output import create_ref_designator_excel_output  # noqa: E402


def _base(label_classes):
    return rp.BaseRender(inner_svg="<g/>", view_box=(100.0, 50.0), label_classes=label_classes)


# ---------------------------------------------------------------------------
# kept_label_positions
# ---------------------------------------------------------------------------

def test_kept_label_positions_drops_excluded_region_labels_including_boundary():
    """守るもの: 領域付きモードでは、「除外」領域の内側（境界線上を含む）のラベルを位置から外す
    （Excel の行を作る `build_ref_designator_region_results()` と同じ。表と図面の件数を一致させる）。"""
    analysis = {'regions': [
        {'id': 0, 'polygon': [(0, 0), (10, 0), (10, 10), (0, 10)], 'name_candidates': [(1.0, 'DRIVE')]},
        {'id': 1, 'polygon': [(20, 0), (30, 0), (30, 10), (20, 10)], 'name_candidates': [(1.0, 'KEEP')]},
    ]}
    labels = [('IN', 5.0, 5.0), ('EDGE', 10.0, 5.0), ('OUT', 25.0, 5.0), ('FAR', 99.0, 99.0)]
    got = lp.kept_label_positions(labels, analysis, {'DRIVE'})
    assert [t for t, _x, _y in got] == ['OUT', 'FAR']
    assert lp.kept_label_positions(labels, analysis, frozenset()) == labels  # 除外なし
    assert lp.kept_label_positions(labels) == labels  # 通常モード


# ---------------------------------------------------------------------------
# sheet_to_file / mark_of_label
# ---------------------------------------------------------------------------

def test_sheet_to_file_uses_the_excel_sheet_name_rule():
    """守るもの: シート名（拡張子なし・31文字まで）でファイルを引く。Summary など図面に対応しないシートは None。"""
    long_name = 'X' * 40 + '.dxf'
    files = ['a.dxf', long_name]
    assert lp.sheet_to_file('a', files) == 'a.dxf'
    assert lp.sheet_to_file('X' * 31, files) == long_name
    for sheet in ('Summary', 'Total', '領域一覧', '抽出ラベル一覧', 'b'):
        assert lp.sheet_to_file(sheet, files) is None


def test_mark_of_label_maps_rows_to_the_three_marks():
    rows = [{'機器符号候補': 'DEF', 'ラベル': 'A'}, {'機器符号候補': 'CAN', 'ラベル': 'B'},
            {'機器符号候補': None, 'ラベル': 'C'}, {'機器符号候補': '', 'ラベル': 'D'}]
    assert lp.mark_of_label(rows) == {'A': 'DEF', 'B': 'CAN', 'C': lp.BLANK_MARK, 'D': lp.BLANK_MARK}


# ---------------------------------------------------------------------------
# build_label_viewer_data
# ---------------------------------------------------------------------------

_POS = [('R10', 1.0, 1.0), ('R10', 2.0, 1.0), ('ABC1', 3.0, 1.0), ('NOTE', 4.0, 1.0), ('GONE', 5.0, 1.0)]
_MARKS = {'R10': 'DEF', 'ABC1': 'CAN', 'NOTE': lp.BLANK_MARK, 'GONE': 'CAN'}
_CLASSES = {('R10', 1.0, 1.0): [['C1', 'fill']], ('R10', 2.0, 1.0): [['C2', 'fill']],
            ('ABC1', 3.0, 1.0): [['C3', 'fill']], ('NOTE', 4.0, 1.0): [['C4', 'fill']]}


def test_viewer_data_colors_only_selected_marks_and_counts_all():
    """守るもの: 色を付けるのは選択した区分のラベルだけ（クラスはラベルの出現ごと）。凡例の件数は選択に関わらず全ラベル。"""
    data = lp.build_label_viewer_data(_base(_CLASSES), 'a.dxf', _POS, _MARKS, {'DEF'})
    assert [(it['m'], it['c']) for it in data['items']] == [('DEF', [['C1', 'fill']]), ('DEF', [['C2', 'fill']])]
    assert {g['mark']: (g['count'], g['placed'], g['selected']) for g in data['groups']} == {
        'DEF': (2, 2, True), 'CAN': (2, 1, False), lp.BLANK_MARK: (1, 1, False)}
    assert data['colors'] == lp.MARK_COLORS
    assert len({lp.MARK_COLORS[m] for m in lp.MARK_ORDER}) == 3  # 区分ごとに別の色
    assert data['unplaced'] == 0


def test_viewer_data_reports_unplaced_labels_without_guessing_a_position():
    """守るもの: 図面に描かれていない（CSS クラスが無い）ラベルは、選択した区分なら件数だけ数え、items には入れない。"""
    data = lp.build_label_viewer_data(_base(_CLASSES), 'a.dxf', _POS, _MARKS, {'CAN', lp.BLANK_MARK})
    assert [it['m'] for it in data['items']] == ['CAN', lp.BLANK_MARK]  # ABC1, NOTE
    assert data['unplaced'] == 1  # GONE
    none_selected = lp.build_label_viewer_data(_base(_CLASSES), 'a.dxf', _POS, _MARKS, set())
    assert none_selected['items'] == [] and none_selected['unplaced'] == 0


# ---------------------------------------------------------------------------
# 領域線（領域付きモード）
# ---------------------------------------------------------------------------

_SQ = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_region_outlines_are_one_line_per_region_in_svg_coordinates():
    """守るもの: 同じ領域に複数の名称（named に同じ id が複数）があっても線は1本。点が3つ未満の多角形は描かない。
    座標は図面の SVG 座標（`region_preview._to_svg` と同じ変換）。"""
    base = rp.render_base_svg(_dxf_bytes())
    named = [{'id': 0, 'polygon': _SQ, 'name': 'A'}, {'id': 0, 'polygon': _SQ, 'name': 'B'},
             {'id': 1, 'polygon': [(0, 0), (1, 1)], 'name': 'BAD'}, {'id': 2, 'polygon': _SQ, 'name': 'C'}]
    out = lp.region_outlines(base, named)
    assert [o['id'] for o in out] == [0, 2]
    assert out[0]['points'] == [list(rp._to_svg(base.matrix, x, y)) for x, y in _SQ]


def test_viewer_data_has_region_lines_only_when_regions_are_given():
    """守るもの: 領域線は named（抽出された領域）を渡したときだけ data に入る。通常モードは空で、色は赤。"""
    base = rp.render_base_svg(_dxf_bytes())  # 座標変換の行列が要るので、実際に描画した結果を使う
    assert lp.build_label_viewer_data(base, 'a.dxf', _POS, _MARKS, {'DEF'})['regions'] == []
    data = lp.build_label_viewer_data(
        base, 'a.dxf', _POS, _MARKS, {'DEF'}, [{'id': 3, 'polygon': _SQ, 'name': 'X'}])
    assert [r['id'] for r in data['regions']] == [3]
    assert data['regionColor'] == lp.REGION_LINE_COLOR == '#ff3030'


# ---------------------------------------------------------------------------
# 画面（app.py）
# ---------------------------------------------------------------------------

def _dxf_bytes():
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (100, 0), (100, 50), (0, 50)], close=True)
    for text, x in (('R10', 10), ('ABC1', 40), ('NOTE', 70)):
        msp.add_text(text, dxfattribs={'insert': (x, 10), 'height': 3})
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode('utf-8')


_ROWS = [{'機器符号候補': 'DEF', 'ラベル': 'R10', '個数': 1},
         {'機器符号候補': 'CAN', 'ラベル': 'ABC1', '個数': 1},
         {'機器符号候補': None, 'ラベル': 'NOTE', '個数': 1}]

_PREVIEW_SCRIPT = '''
import sys
sys.path.insert(0, {project_root!r})
import base64
import streamlit as st
from app import _render_results_section
from model import region_preview
from view import label_viewer

class _UF:
    name = 'a.dxf'
    def getvalue(self):
        return base64.b64decode({dxf_b64!r})

calls = st.session_state.setdefault('_viewer_calls', [])
label_viewer.mount_viewer = lambda **kw: calls.append(kw['data'])
base = region_preview.render_base_svg(_UF().getvalue())
st.session_state['excel_result'] = base64.b64decode({excel_b64!r})
st.session_state['output_filename'] = 'extracted_labels.xlsx'
st.session_state['is_region_mode'] = False
st.session_state['ref_results_summary'] = {{'a.dxf': {{}}}}
st.session_state['extract_results_full'] = {{'a.dxf': dict({full_extra!r}, rows={rows!r})}}
st.session_state['extract_label_positions'] = {{'a.dxf': list(base.label_classes)}}
_render_results_section(uploaded_files={files})
'''


def _run_preview(with_file=True, named=None):
    ref_final = {'a.dxf': {
        'rows': _ROWS, 'total_in_frame': 3, 'ref_designator_count': 2, 'frames': 1, 'warning': None,
        'main_drawing_number': 'EE1234-500-01A', 'source_drawing_number': None, 'title': 'T', 'subtitle': None}}
    excel = create_ref_designator_excel_output(ref_final, sort_option='asc')
    script = _PREVIEW_SCRIPT.format(
        project_root=PROJECT_ROOT, dxf_b64=base64.b64encode(_dxf_bytes()).decode('ascii'),
        excel_b64=base64.b64encode(excel).decode('ascii'), rows=_ROWS,
        full_extra={} if named is None else {'named': named},
        files='[_UF()]' if with_file else '[]')
    at = AppTest.from_string(script)
    at.run()
    return at


def _toggles(at):
    return [t for t in at.toggle if t.key == 'extract_preview_viewer_a']


def test_preview_shows_the_viewer_toggle_only_on_a_drawing_sheet():
    """守るもの: 「図面を表示」トグルは図面に対応するシートだけに出る（Summary・抽出ラベル一覧には出ない）。
    OFF の間は viewer を描かない。"""
    at = _run_preview()
    assert at.exception.len == 0
    assert _toggles(at) == [] and at.session_state['_viewer_calls'] == []  # Summary
    at.selectbox(key='extract_preview_sheet').select('a').run()
    assert at.exception.len == 0
    assert len(_toggles(at)) == 1 and _toggles(at)[0].value is False
    assert at.session_state['_viewer_calls'] == []


def test_preview_viewer_follows_the_candidate_selection():
    """守るもの: ON にすると viewer に、「機器符号候補」の選択（DEF/CAN/（空欄））に当たるラベルだけが渡る。"""
    at = _run_preview()
    at.selectbox(key='extract_preview_sheet').select('a').run()
    _toggles(at)[0].set_value(True).run()
    assert at.exception.len == 0
    data = at.session_state['_viewer_calls'][-1]
    assert {it['m'] for it in data['items']} == {'DEF', 'CAN', lp.BLANK_MARK}  # 既定は全選択
    assert data['unplaced'] == 0
    pills = [p for p in at.pills if p.key.startswith('extract_cand_filter_a:')]
    assert len(pills) == 1
    pills[0].set_value(['DEF']).run()
    assert at.exception.len == 0
    assert _toggles(at)[0].value is True  # 選択を変えても ON のまま
    data = at.session_state['_viewer_calls'][-1]
    assert {it['m'] for it in data['items']} == {'DEF'}
    assert {g['mark']: g['selected'] for g in data['groups']} == {
        'DEF': True, 'CAN': False, lp.BLANK_MARK: False}


def test_preview_without_uploaded_file_explains_instead_of_toggle():
    """守るもの: アップロードが残っていないファイルでは、トグルの代わりに理由を示す（図面は出せない）。"""
    at = _run_preview(with_file=False)
    at.selectbox(key='extract_preview_sheet').select('a').run()
    assert at.exception.len == 0
    assert _toggles(at) == []
    assert any('図面を表示できません' in c.value for c in at.caption)


def test_preview_viewer_draws_region_lines_only_in_region_mode():
    """守るもの: 領域付きモードの結果（`named` あり）では viewer に領域線が渡り、通常モードの結果（`named` なし）では渡らない。"""
    at = _run_preview()  # 通常モード
    at.selectbox(key='extract_preview_sheet').select('a').run()
    _toggles(at)[0].set_value(True).run()
    assert at.session_state['_viewer_calls'][-1]['regions'] == []

    at = _run_preview(named=[{'id': 0, 'polygon': _SQ, 'name': 'A'}, {'id': 0, 'polygon': _SQ, 'name': 'B'}])
    at.selectbox(key='extract_preview_sheet').select('a').run()
    _toggles(at)[0].set_value(True).run()
    assert at.exception.len == 0
    assert [r['id'] for r in at.session_state['_viewer_calls'][-1]['regions']] == [0]
