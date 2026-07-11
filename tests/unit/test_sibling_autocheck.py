"""utils/ref_designator.py の兄弟ラベル連動（sibling_key / propagate_selection_all_files）の単体テスト。

組み合わせ表:
  - sibling_key: 末尾数字1桁/2桁（対象）、3桁以上・末尾英字・数字のみ・括弧付き（対象外）、
    全角/半角（NFKC正規化で同一キー）
  - propagate_selection_all_files: 採用の連動（ON伝播）、解除の連動（OFF伝播＝逆方向）、
    連動対象外ラベルの非干渉、全角/半角混在での連動、プレフィックス不一致の非連動、
    全ファイル横断の連動（兄弟・同一ラベル）、他ファイルに無いラベルを追加しないこと
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils import ref_designator


class TestSiblingKey(unittest.TestCase):
    def test_trailing_one_and_two_digits_share_key(self):
        self.assertEqual(ref_designator.sibling_key('CN1'), 'CN')
        self.assertEqual(ref_designator.sibling_key('CN2'), 'CN')
        self.assertEqual(ref_designator.sibling_key('CN10'), 'CN')

    def test_prefix_mismatch_gives_different_key(self):
        self.assertEqual(ref_designator.sibling_key('R5'), 'R')
        self.assertEqual(ref_designator.sibling_key('RA5'), 'RA')
        self.assertNotEqual(
            ref_designator.sibling_key('R5'), ref_designator.sibling_key('RA5'))

    def test_hyphenated_prefix(self):
        self.assertEqual(ref_designator.sibling_key('CN-IF3'), 'CN-IF')

    def test_three_or_more_trailing_digits_excluded(self):
        self.assertIsNone(ref_designator.sibling_key('CB001'))
        self.assertIsNone(ref_designator.sibling_key('CN100'))

    def test_non_digit_tail_excluded(self):
        self.assertIsNone(ref_designator.sibling_key('X14A'))
        self.assertIsNone(ref_designator.sibling_key('FB'))
        self.assertIsNone(ref_designator.sibling_key('R10(2.2K)'))

    def test_digits_only_excluded(self):
        self.assertIsNone(ref_designator.sibling_key('10'))
        self.assertIsNone(ref_designator.sibling_key('1'))

    def test_fullwidth_normalized_to_same_key(self):
        # 全角 ＣＮ１ は NFKC 正規化により半角 CN1 と同じキーになる
        self.assertEqual(ref_designator.sibling_key('ＣＮ１'), 'CN')
        self.assertEqual(
            ref_designator.sibling_key('ＣＮ１'), ref_designator.sibling_key('CN10'))


class TestPropagateSelectionAllFiles(unittest.TestCase):
    def test_adopt_propagates_to_siblings(self):
        state = {'a.dxf': {'CN1': False, 'CN2': False, 'CN10': False, 'X14A': False}}
        ref_designator.propagate_selection_all_files(state, 'CN1', True)
        self.assertEqual(
            state['a.dxf'],
            {'CN1': True, 'CN2': True, 'CN10': True, 'X14A': False})

    def test_unadopt_propagates_to_siblings(self):
        # 逆方向: 1つ解除すると兄弟も解除される
        state = {'a.dxf': {'CN1': True, 'CN2': True, 'CN10': True, 'X14A': True}}
        ref_designator.propagate_selection_all_files(state, 'CN10', False)
        self.assertEqual(
            state['a.dxf'],
            {'CN1': False, 'CN2': False, 'CN10': False, 'X14A': True})

    def test_non_sibling_label_toggles_alone(self):
        state = {'a.dxf': {'CN1': False, 'X14A': False, 'CB001': False, 'CB002': False}}
        ref_designator.propagate_selection_all_files(state, 'X14A', True)
        self.assertEqual(
            state['a.dxf'],
            {'CN1': False, 'X14A': True, 'CB001': False, 'CB002': False})
        # 末尾3桁は sibling_key=None → CB001 を採用しても CB002 は連動しない
        ref_designator.propagate_selection_all_files(state, 'CB001', True)
        self.assertEqual(
            state['a.dxf'],
            {'CN1': False, 'X14A': True, 'CB001': True, 'CB002': False})

    def test_prefix_mismatch_not_propagated(self):
        state = {'a.dxf': {'R5': False, 'RA5': False, 'R12': False}}
        ref_designator.propagate_selection_all_files(state, 'R5', True)
        self.assertEqual(state['a.dxf'], {'R5': True, 'RA5': False, 'R12': True})

    def test_fullwidth_and_halfwidth_linked(self):
        # 全角表記のラベルも半角の兄弟と連動する（表示テキストは原文のまま）
        state = {'a.dxf': {'ＣＮ１': False, 'CN10': False}}
        ref_designator.propagate_selection_all_files(state, 'CN10', True)
        self.assertEqual(state['a.dxf'], {'ＣＮ１': True, 'CN10': True})

    def test_siblings_propagate_across_files(self):
        # 兄弟連動はファイルを跨いで全ファイルに伝播する
        state = {
            'a.dxf': {'CN1': False, 'X14A': False},
            'b.dxf': {'CN2': False, 'CN10': False, 'R5': False},
        }
        ref_designator.propagate_selection_all_files(state, 'CN1', True)
        self.assertEqual(state['a.dxf'], {'CN1': True, 'X14A': False})
        self.assertEqual(state['b.dxf'], {'CN2': True, 'CN10': True, 'R5': False})

    def test_same_label_syncs_across_files_even_without_sibling_key(self):
        # sibling_key=None のラベルでも、同一ラベルは全ファイルで同期する
        state = {
            'a.dxf': {'X14A': False, 'CN1': False},
            'b.dxf': {'X14A': False, 'CB001': False},
        }
        ref_designator.propagate_selection_all_files(state, 'X14A', True)
        self.assertEqual(state['a.dxf'], {'X14A': True, 'CN1': False})
        self.assertEqual(state['b.dxf'], {'X14A': True, 'CB001': False})
        # 解除も同期する
        ref_designator.propagate_selection_all_files(state, 'X14A', False)
        self.assertEqual(state['a.dxf']['X14A'], False)
        self.assertEqual(state['b.dxf']['X14A'], False)

    def test_fullwidth_same_label_syncs_across_files(self):
        # 別ファイルの全角表記の同一ラベルも同期する
        state = {
            'a.dxf': {'X14A': False},
            'b.dxf': {'Ｘ１４Ａ': False},
        }
        ref_designator.propagate_selection_all_files(state, 'X14A', True)
        self.assertEqual(state['b.dxf'], {'Ｘ１４Ａ': True})

    def test_absent_label_not_added_to_other_files(self):
        # 他ファイルに存在しないラベルは追加されない
        state = {
            'a.dxf': {'CN1': False},
            'b.dxf': {'R5': False},
        }
        ref_designator.propagate_selection_all_files(state, 'CN1', True)
        self.assertEqual(state['b.dxf'], {'R5': False})


if __name__ == '__main__':
    unittest.main()
