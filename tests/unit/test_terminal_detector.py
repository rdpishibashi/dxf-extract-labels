"""model/terminal_detector.py の単体テスト（モデル層のみ、Streamlit UI 非依存）。

組み合わせ表:
  - _TB_LABEL_PATTERN: TB+英大文字/数字（候補）/ TB+漢字・TB+スペース（非候補）
  - _build_rect_candidates: 単一CIRCLE橋渡し矩形 / 複数CIRCLE橋渡し矩形 /
    CIRCLE橋渡しが無いLINEのみの矩形（除外）
  - _label_rect_distances / _match_labels_to_rects: ラベルが矩形の上（primary）/
    下（secondary）/ 隣接2矩形の上下衝突（優先度によるグリーディ解消）/
    対応する矩形が無い（unmatched）/ 90°回転時の左右優先
  - _collect_digits: 半角数字 / 全角数字混在
  - _format_numbers_with_counts: 重複なし / 重複あり（件数表示）
  - analyze_dxf_terminals: 対象ファイル判定（タイトル一致・TB COMPONENT除外）・
    候補パターン非該当ラベルの除外を含む end-to-end（合成DXF、実データでの
    検証は tests/regression 側で実施）
  - build_terminal_rows: 単一ファイル / 複数ファイルにまたがる端子台の統合
    （端子No.の件数表示・図番のABC順結合）/ 対象外ファイルの除外 /
    矩形が見つからない候補ラベルの「端子検出不可」行への振り分け
"""
import os
import sys
import unittest

import ezdxf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from model import terminal_detector as td


LW, COLOR = td.TB_LINE_WEIGHT, td.TB_LINE_COLOR
CLW, CCOLOR = td.TB_CIRCLE_WEIGHT, td.TB_CIRCLE_COLOR


class TestTbLabelPattern(unittest.TestCase):
    def test_tb_plus_digits_matches(self):
        self.assertTrue(td._TB_LABEL_PATTERN.match('TB001'))

    def test_tb_plus_uppercase_letters_and_digits_matches(self):
        self.assertTrue(td._TB_LABEL_PATTERN.match('TBN241'))
        self.assertTrue(td._TB_LABEL_PATTERN.match('TBLSMTP'))

    def test_trailing_content_after_core_is_ignored(self):
        self.assertTrue(td._TB_LABEL_PATTERN.match('TB001 (27A)'))

    def test_tb_plus_kanji_does_not_match(self):
        self.assertIsNone(td._TB_LABEL_PATTERN.match('TB取付板'))

    def test_tb_plus_space_does_not_match(self):
        self.assertIsNone(td._TB_LABEL_PATTERN.match('TB COMPONENT'))

    def test_bare_tb_does_not_match(self):
        self.assertIsNone(td._TB_LABEL_PATTERN.match('TB'))


class TestBuildRectCandidates(unittest.TestCase):
    def test_single_bridge_rectangle_detected(self):
        # 10x10の矩形。左辺(x=0)を2分割し、CIRCLEで橋渡しする。
        lines = [
            (0, 0, 0, 4),      # 左辺 下側
            (0, 6, 0, 10),     # 左辺 上側
            (10, 0, 10, 10),   # 右辺（連続）
            (0, 0, 10, 0),     # 下辺
            (0, 10, 10, 10),   # 上辺
        ]
        circles = [(0, 5, 1.0)]  # 左辺のギャップ(4-6)を橋渡し
        rects = td._build_rect_candidates(lines, circles)
        self.assertEqual(len(rects), 1)
        r = rects[0]
        self.assertAlmostEqual(r['xl'], 0)
        self.assertAlmostEqual(r['xr'], 10)
        self.assertAlmostEqual(r['y0'], 0)
        self.assertAlmostEqual(r['y1'], 10)

    def test_multiple_bridges_single_rectangle(self):
        # 縦に長い矩形、複数箇所で橋渡し（TB004のような複数端子の矩形を模す）
        lines = [
            (0, 0, 0, 4), (0, 6, 0, 14), (0, 16, 0, 20),   # 左辺 3分割
            (10, 0, 10, 20),                               # 右辺
            (0, 0, 10, 0), (0, 20, 10, 20),                # 下辺・上辺
        ]
        circles = [(0, 5, 1.0), (0, 15, 1.0)]
        rects = td._build_rect_candidates(lines, circles)
        self.assertEqual(len(rects), 1)
        self.assertAlmostEqual(rects[0]['y1'], 20)

    def test_line_only_rectangle_without_bridge_is_excluded(self):
        # CIRCLE橋渡しが無い、閉じたLINEのみの矩形は対象外
        lines = [
            (0, 0, 0, 10), (10, 0, 10, 10),
            (0, 0, 10, 0), (0, 10, 10, 10),
        ]
        rects = td._build_rect_candidates(lines, circles=[])
        self.assertEqual(rects, [])

    def test_bridge_without_full_side_coverage_is_excluded(self):
        # 円で橋渡しされても、4辺のうち1辺が欠けていれば矩形として扱わない
        lines = [
            (0, 0, 0, 4), (0, 6, 0, 10),   # 左辺（分割）
            (10, 0, 10, 10),               # 右辺
            (0, 0, 10, 0),                 # 下辺のみ（上辺が無い）
        ]
        circles = [(0, 5, 1.0)]
        rects = td._build_rect_candidates(lines, circles)
        self.assertEqual(rects, [])


class TestLabelRectMatching(unittest.TestCase):
    def _rect(self, xl, xr, y0, y1):
        return {'xl': xl, 'xr': xr, 'y0': y0, 'y1': y1}

    def test_label_above_rectangle_is_primary(self):
        rect = self._rect(0, 10, 0, 10)
        primary, secondary = td._label_rect_distances(None, 5, 14, 0, 10, 0, 10, 1.0)
        self.assertEqual(primary, 4)   # 14 - 10
        self.assertEqual(secondary, -14)  # 0 - 14

    def test_label_below_rectangle_is_secondary(self):
        primary, secondary = td._label_rect_distances(None, 5, -4, 0, 10, 0, 10, 1.0)
        self.assertEqual(secondary, 4)  # 0 - (-4)

    def test_label_outside_x_range_returns_none(self):
        primary, secondary = td._label_rect_distances(None, 50, 14, 0, 10, 0, 10, 1.0)
        self.assertIsNone(primary)
        self.assertIsNone(secondary)

    def test_adjacent_rectangles_tiebreak_resolved_by_priority(self):
        """EE6492-039-38A.dxf で実際に発生した衝突を単純化して再現する:
        隣接する2矩形の間にラベルAがあり、ラベルAは上の矩形へのsecondary距離が
        より近い(3.5)が、下の矩形へのprimary距離(4.0)を優先すべき。ラベルBは
        上の矩形へのprimary候補（4.0、唯一の候補）しか持たない。
        両ラベルとも正しい矩形に割り当てられることを確認する。"""
        lower = self._rect(572.41, 582.41, 245.0, 250.0)   # 下の矩形
        upper = self._rect(572.41, 582.41, 257.5, 262.5)   # 上の矩形
        rects = [lower, upper]
        # ラベルA: y=254.0 (下矩形からprimary=4.0 / 上矩形へsecondary=3.5)
        # ラベルB: y=266.5 (上矩形からprimary=4.0のみ)
        tb_labels = [
            ('TBP044', 577.41, 254.0, 0),
            ('TBN241', 577.41, 266.5, 0),
        ]
        matches, unmatched = td._match_labels_to_rects(tb_labels, rects, rotated_mode=None)
        self.assertEqual(unmatched, [])
        self.assertEqual(matches[0], 0)  # TBP044 -> 下の矩形(lower, idx0)
        self.assertEqual(matches[1], 1)  # TBN241 -> 上の矩形(upper, idx1)

    def test_label_with_no_nearby_rectangle_is_unmatched(self):
        rects = [self._rect(0, 10, 0, 10)]
        tb_labels = [('TBFAR', 500, 500, 0)]  # 遠く離れた位置
        matches, unmatched = td._match_labels_to_rects(tb_labels, rects, rotated_mode=None)
        self.assertEqual(matches, {})
        self.assertEqual(unmatched, [0])

    def test_rotated_right_prefers_label_to_the_right(self):
        rect = self._rect(0, 10, 0, 10)
        # rotated_mode='right' のとき、primary = lx - xr（矩形の右側にあるほど近い）
        primary, secondary = td._label_rect_distances('right', 14, 5, 0, 10, 0, 10, 1.0)
        self.assertEqual(primary, 4)  # 14 - 10
        primary2, secondary2 = td._label_rect_distances('left', -4, 5, 0, 10, 0, 10, 1.0)
        self.assertEqual(primary2, 4)  # 0 - (-4)


class TestDetectRotationMode(unittest.TestCase):
    def test_no_rotation_returns_none(self):
        texts = [('A', 0, 0, 0), ('B', 1, 1, 0)]
        self.assertIsNone(td._detect_rotation_mode(texts))

    def test_plus90_majority_returns_right(self):
        texts = [('A', 0, 0, 90), ('B', 1, 1, 90), ('C', 2, 2, 0)]
        self.assertEqual(td._detect_rotation_mode(texts), 'right')

    def test_minus90_majority_returns_left(self):
        texts = [('A', 0, 0, -90), ('B', 1, 1, -90), ('C', 2, 2, 0)]
        self.assertEqual(td._detect_rotation_mode(texts), 'left')


class TestCollectDigits(unittest.TestCase):
    def test_halfwidth_and_fullwidth_digits(self):
        rect = {'xl': 0, 'xr': 10, 'y0': 0, 'y1': 10}
        texts = [
            ('1', 5, 5, 0),
            ('２', 3, 3, 0),      # 全角
            ('TB001', 5, 5, 0),  # 数字のみでないので除外
            ('3', 500, 500, 0),  # 矩形の外なので除外
        ]
        digits = td._collect_digits(texts, rect)
        self.assertEqual(sorted(digits), [1, 2])


class TestFormatNumbersWithCounts(unittest.TestCase):
    def test_no_duplicates_shows_plain_numbers(self):
        self.assertEqual(td._format_numbers_with_counts([3, 1, 2]), '1, 2, 3')

    def test_duplicate_shows_count_in_parens(self):
        self.assertEqual(td._format_numbers_with_counts([1, 2, 3, 7, 7]), '1, 2, 3, 7(2)')

    def test_multiple_duplicates(self):
        self.assertEqual(td._format_numbers_with_counts([5, 5, 5, 2, 2]), '2(2), 5(3)')


class TestAnalyzeDxfTerminalsEndToEnd(unittest.TestCase):
    """合成DXFによる end-to-end 確認。実データでの検証は
    tests/regression/test_terminal_detector.py 側で行う。"""

    def _make_doc_with_titleblock(self, title, subtitle=None):
        doc = ezdxf.new()
        msp = doc.modelspace()
        msp.add_text('TITLE', dxfattribs={'insert': (0, 100)})
        msp.add_text(title, dxfattribs={'insert': (20, 100)})
        if subtitle:
            msp.add_text(subtitle, dxfattribs={'insert': (20, 90)})
        return doc, msp

    def _add_tb_rectangle(self, msp, x0, y_label, label_text, digit_texts):
        """label_textの下(primary=上)に矩形を1つ追加する。
        矩形は x=[x0, x0+10], y=[y_label-14, y_label-4]（labelからprimary距離4）。"""
        y1 = y_label - 4
        y0 = y1 - 10
        xl, xr = x0, x0 + 10
        msp.add_line((xl, y0), (xl, y0 + 4), dxfattribs={'lineweight': LW, 'color': COLOR})
        msp.add_line((xl, y0 + 6), (xl, y1), dxfattribs={'lineweight': LW, 'color': COLOR})
        msp.add_line((xr, y0), (xr, y1), dxfattribs={'lineweight': LW, 'color': COLOR})
        msp.add_line((xl, y0), (xr, y0), dxfattribs={'lineweight': LW, 'color': COLOR})
        msp.add_line((xl, y1), (xr, y1), dxfattribs={'lineweight': LW, 'color': COLOR})
        msp.add_circle((xl, y0 + 5), 1.0, dxfattribs={'lineweight': CLW, 'color': CCOLOR})
        msp.add_text(label_text, dxfattribs={'insert': (xl + 3, y_label)})
        cy = (y0 + y1) / 2
        for i, dt in enumerate(digit_texts):
            msp.add_text(dt, dxfattribs={'insert': (xl + 3, cy + i * 0.1)})

    def test_target_file_with_matching_rectangle(self):
        doc, msp = self._make_doc_with_titleblock('UNIT内結線図', 'TEST SUBTITLE')
        self._add_tb_rectangle(msp, x0=0, y_label=20, label_text='TB001', digit_texts=['1', '2'])
        path = '/tmp/_test_terminal_target.dxf'
        doc.saveas(path)
        try:
            result = td.analyze_dxf_terminals(path, original_filename='sample.dxf')
        finally:
            os.remove(path)

        self.assertTrue(result['is_target'])
        self.assertEqual(result['unmatched_labels'], [])
        self.assertEqual(len(result['entries']), 1)
        self.assertEqual(result['entries'][0]['label'], 'TB001')
        self.assertEqual(result['entries'][0]['numbers'], [1, 2])

    def test_tb_rectangle_in_paper_space_layout_is_detected_when_model_space_empty(self):
        """Model Space が完全に空で、端子台矩形一式が Paper Space レイアウトに
        配置されている図面でも検出できること。Model Space と Paper Space は
        独立した座標系のため、Model Space に何らかの内容がある場合はそちらを
        優先し他レイアウトを見に行かない（`_collect_geometry` は Model Space
        が完全に空の場合のみ他レイアウトを試す。`EE6892-455B.dxf` で
        Paper Space 由来の図面枠を Model Space のラベルに誤って適用し
        `CB001`等が消える不具合が発生したため、レイアウトをまたいで座標系を
        混在させない設計にした。2026-07-14）。"""
        doc = ezdxf.new()
        paper_space = doc.layout('Layout1')
        paper_space.add_text('TITLE', dxfattribs={'insert': (0, 100)})
        paper_space.add_text('UNIT内結線図', dxfattribs={'insert': (20, 100)})
        paper_space.add_text('TEST SUBTITLE', dxfattribs={'insert': (20, 90)})
        self._add_tb_rectangle(paper_space, x0=0, y_label=20, label_text='TB001',
                                digit_texts=['1', '2'])
        path = '/tmp/_test_terminal_paperspace.dxf'
        doc.saveas(path)
        try:
            result = td.analyze_dxf_terminals(path, original_filename='sample.dxf')
        finally:
            os.remove(path)

        self.assertTrue(result['is_target'])
        self.assertEqual(len(result['entries']), 1)
        self.assertEqual(result['entries'][0]['label'], 'TB001')
        self.assertEqual(result['entries'][0]['numbers'], [1, 2])

    def test_model_space_content_is_not_filtered_by_other_layout_geometry(self):
        """Model Space に内容がある場合は、他レイアウトに矩形やラベルが
        あっても Model Space の結果のみを使う（レイアウトをまたいだ座標系の
        混在を防ぐ）。Model Space 側の TB ラベルには対応する矩形が無いため
        unmatched 扱いになるが、Paper Space側の無関係な矩形を誤って
        流用しない。"""
        doc, msp = self._make_doc_with_titleblock('UNIT内結線図')
        msp.add_text('TB999', dxfattribs={'insert': (500, 500)})  # 対応する矩形なし
        paper_space = doc.layout('Layout1')
        self._add_tb_rectangle(paper_space, x0=0, y_label=20, label_text='TB001',
                                digit_texts=['1'])
        path = '/tmp/_test_terminal_mixed_layout.dxf'
        doc.saveas(path)
        try:
            result = td.analyze_dxf_terminals(path, original_filename='sample.dxf')
        finally:
            os.remove(path)

        self.assertEqual(result['entries'], [])
        self.assertEqual(result['unmatched_labels'], ['TB999'])

    def test_non_target_title_is_skipped(self):
        doc, msp = self._make_doc_with_titleblock('SYSTEM_I/F_BOX_本体')
        self._add_tb_rectangle(msp, x0=0, y_label=20, label_text='TB001', digit_texts=['1'])
        path = '/tmp/_test_terminal_nontarget.dxf'
        doc.saveas(path)
        try:
            result = td.analyze_dxf_terminals(path, original_filename='sample.dxf')
        finally:
            os.remove(path)

        self.assertFalse(result['is_target'])
        self.assertEqual(result['entries'], [])

    def test_tb_component_subtitle_is_excluded(self):
        doc, msp = self._make_doc_with_titleblock('UNIT内結線図', 'TB COMPONENT')
        self._add_tb_rectangle(msp, x0=0, y_label=20, label_text='TB001', digit_texts=['1'])
        path = '/tmp/_test_terminal_tbcomponent.dxf'
        doc.saveas(path)
        try:
            result = td.analyze_dxf_terminals(path, original_filename='sample.dxf')
        finally:
            os.remove(path)

        self.assertFalse(result['is_target'])
        self.assertEqual(result['entries'], [])

    def test_kanji_suffix_label_is_not_a_candidate(self):
        """TBの直後が漢字（例: TB取付板）は候補パターンに一致せず、矩形が
        近くに無くても unmatched_labels にすら現れない（候補選定の時点で
        除外されるため）。"""
        doc, msp = self._make_doc_with_titleblock('UNIT内結線図')
        msp.add_text('TB取付板', dxfattribs={'insert': (100, 100)})
        path = '/tmp/_test_terminal_kanji_suffix.dxf'
        doc.saveas(path)
        try:
            result = td.analyze_dxf_terminals(path, original_filename='sample.dxf')
        finally:
            os.remove(path)

        self.assertTrue(result['is_target'])
        self.assertEqual(result['entries'], [])
        self.assertEqual(result['unmatched_labels'], [])

    def test_duplicate_digit_across_rectangles_is_preserved(self):
        """1ファイル内で同一ラベルが2つの矩形にまたがり、同じ端子番号が
        重複して現れる場合、analyze_dxf_terminals の時点では重複除去せず
        保持する（件数表示は build_terminal_rows 側の責務）。"""
        doc, msp = self._make_doc_with_titleblock('UNIT内結線図')
        self._add_tb_rectangle(msp, x0=0, y_label=20, label_text='TB001', digit_texts=['1'])
        self._add_tb_rectangle(msp, x0=100, y_label=20, label_text='TB001', digit_texts=['1'])
        path = '/tmp/_test_terminal_dup_digit.dxf'
        doc.saveas(path)
        try:
            result = td.analyze_dxf_terminals(path, original_filename='sample.dxf')
        finally:
            os.remove(path)

        self.assertEqual(len(result['entries']), 1)
        self.assertEqual(result['entries'][0]['numbers'], [1, 1])


class TestBuildTerminalRows(unittest.TestCase):
    def test_single_file_row(self):
        results = {
            'a.dxf': {
                'is_target': True, 'drawing_number': 'EE0001',
                'entries': [{'label': 'TB001', 'numbers': [1, 2]}],
            },
        }
        rows = td.build_terminal_rows(results)
        self.assertEqual(rows, [{'端子台': 'TB001', '端子No.': '1, 2', '図番': 'EE0001'}])

    def test_non_target_file_is_excluded(self):
        results = {
            'a.dxf': {
                'is_target': True, 'drawing_number': 'EE0001',
                'entries': [{'label': 'TB001', 'numbers': [1]}],
            },
            'b.dxf': {'is_target': False, 'entries': []},
        }
        rows = td.build_terminal_rows(results)
        self.assertEqual(len(rows), 1)

    def test_same_label_across_files_is_merged_into_one_row(self):
        """同じ端子台名が複数ファイルにまたがる場合、端子番号・図番を
        1行に統合する（端子台でファイル横断ユニーク）。"""
        results = {
            'b.dxf': {
                'is_target': True, 'drawing_number': 'EE0002',
                'entries': [{'label': 'TBN241', 'numbers': [5, 6]}],
            },
            'a.dxf': {
                'is_target': True, 'drawing_number': 'EE0001',
                'entries': [{'label': 'TBN241', 'numbers': [1, 2]}],
            },
        }
        rows = td.build_terminal_rows(results)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['端子台'], 'TBN241')
        self.assertEqual(rows[0]['端子No.'], '1, 2, 5, 6')
        # 図番はABC順（EE0001, EE0002）
        self.assertEqual(rows[0]['図番'], 'EE0001, EE0002')

    def test_duplicate_number_across_files_shows_count(self):
        results = {
            'a.dxf': {
                'is_target': True, 'drawing_number': 'EE0001',
                'entries': [{'label': 'TB001', 'numbers': [1, 2]}],
            },
            'b.dxf': {
                'is_target': True, 'drawing_number': 'EE0002',
                'entries': [{'label': 'TB001', 'numbers': [1]}],
            },
        }
        rows = td.build_terminal_rows(results)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['端子No.'], '1(2), 2')

    def test_rows_sorted_by_label_abc_order(self):
        results = {
            'a.dxf': {
                'is_target': True, 'drawing_number': 'EE0001',
                'entries': [
                    {'label': 'TB002', 'numbers': [1]},
                    {'label': 'TB001', 'numbers': [1]},
                ],
            },
        }
        rows = td.build_terminal_rows(results)
        self.assertEqual([r['端子台'] for r in rows], ['TB001', 'TB002'])

    def test_unmatched_label_appended_after_blank_row(self):
        """矩形が見つからなかった候補ラベルは、末尾に空行を挟んで
        '端子検出不可' 行として追加される（端子No.列にラベル・図番列に図番）。"""
        results = {
            'a.dxf': {
                'is_target': True, 'drawing_number': 'EE0001',
                'entries': [{'label': 'TB001', 'numbers': [1]}],
                'unmatched_labels': ['TBZZZ'],
            },
        }
        rows = td.build_terminal_rows(results)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]['端子台'], 'TB001')
        self.assertEqual(rows[1], {'端子台': '', '端子No.': '', '図番': ''})
        self.assertEqual(rows[2], {
            '端子台': '端子検出不可', '端子No.': 'TBZZZ', '図番': 'EE0001',
        })

    def test_no_blank_row_when_no_unmatched_labels(self):
        results = {
            'a.dxf': {
                'is_target': True, 'drawing_number': 'EE0001',
                'entries': [{'label': 'TB001', 'numbers': [1]}],
                'unmatched_labels': [],
            },
        }
        rows = td.build_terminal_rows(results)
        self.assertEqual(len(rows), 1)

    def test_unmatched_label_across_files_merges_drawing_numbers(self):
        results = {
            'b.dxf': {
                'is_target': True, 'drawing_number': 'EE0002',
                'entries': [], 'unmatched_labels': ['TBZZZ'],
            },
            'a.dxf': {
                'is_target': True, 'drawing_number': 'EE0001',
                'entries': [], 'unmatched_labels': ['TBZZZ'],
            },
        }
        rows = td.build_terminal_rows(results)
        # 空行 + 統合された1行のみ
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]['図番'], 'EE0001, EE0002')

    def test_multiple_unmatched_labels_sorted_abc_order(self):
        results = {
            'a.dxf': {
                'is_target': True, 'drawing_number': 'EE0001',
                'entries': [], 'unmatched_labels': ['TBZZZ', 'TBAAA'],
            },
        }
        rows = td.build_terminal_rows(results)
        unmatched_rows = rows[1:]  # rows[0] は空行
        self.assertEqual([r['端子No.'] for r in unmatched_rows], ['TBAAA', 'TBZZZ'])

    def test_non_target_file_unmatched_labels_excluded(self):
        results = {
            'a.dxf': {'is_target': False, 'entries': [], 'unmatched_labels': ['TBZZZ']},
        }
        rows = td.build_terminal_rows(results)
        self.assertEqual(rows, [])


if __name__ == '__main__':
    unittest.main()
