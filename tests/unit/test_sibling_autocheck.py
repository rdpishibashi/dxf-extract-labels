"""utils/ref_designator.py の兄弟ラベル連動（sibling_key / propagate_sibling_selection）の単体テスト。

組み合わせ表:
  - sibling_key: 末尾数字1桁/2桁（対象）、3桁以上・末尾英字・数字のみ・括弧付き（対象外）、
    全角/半角（NFKC正規化で同一キー）
  - propagate_sibling_selection: 採用の連動（ON伝播）、解除の連動（OFF伝播＝逆方向）、
    連動対象外ラベルの非干渉、全角/半角混在での連動、プレフィックス不一致の非連動
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


class TestPropagateSiblingSelection(unittest.TestCase):
    def test_adopt_propagates_to_siblings(self):
        checked = {'CN1': False, 'CN2': False, 'CN10': False, 'X14A': False}
        ref_designator.propagate_sibling_selection(checked, 'CN1', True)
        self.assertEqual(
            checked, {'CN1': True, 'CN2': True, 'CN10': True, 'X14A': False})

    def test_unadopt_propagates_to_siblings(self):
        # 逆方向: 1つ解除すると兄弟も解除される
        checked = {'CN1': True, 'CN2': True, 'CN10': True, 'X14A': True}
        ref_designator.propagate_sibling_selection(checked, 'CN10', False)
        self.assertEqual(
            checked, {'CN1': False, 'CN2': False, 'CN10': False, 'X14A': True})

    def test_non_sibling_label_toggles_alone(self):
        checked = {'CN1': False, 'X14A': False, 'CB001': False, 'CB002': False}
        ref_designator.propagate_sibling_selection(checked, 'X14A', True)
        self.assertEqual(
            checked, {'CN1': False, 'X14A': True, 'CB001': False, 'CB002': False})
        # 末尾3桁は sibling_key=None → CB001 を採用しても CB002 は連動しない
        ref_designator.propagate_sibling_selection(checked, 'CB001', True)
        self.assertEqual(
            checked, {'CN1': False, 'X14A': True, 'CB001': True, 'CB002': False})

    def test_prefix_mismatch_not_propagated(self):
        checked = {'R5': False, 'RA5': False, 'R12': False}
        ref_designator.propagate_sibling_selection(checked, 'R5', True)
        self.assertEqual(checked, {'R5': True, 'RA5': False, 'R12': True})

    def test_fullwidth_and_halfwidth_linked(self):
        # 全角表記のラベルも半角の兄弟と連動する（表示テキストは原文のまま）
        checked = {'ＣＮ１': False, 'CN10': False}
        ref_designator.propagate_sibling_selection(checked, 'CN10', True)
        self.assertEqual(checked, {'ＣＮ１': True, 'CN10': True})


if __name__ == '__main__':
    unittest.main()
