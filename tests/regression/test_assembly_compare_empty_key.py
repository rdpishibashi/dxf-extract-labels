"""結線図-組立図比較: 括弧で始まるラベルが空キーに合算されない（bugfix 2026-09-26）。

以前の不具合: `_compare_key()` は "(" 以降を除去するため、`(BU)`・`(BK)` 等の
括弧で始まるラベル（電線色などの補足）はすべて空文字列になり、比較シート上で
無関係なラベル群が1行の空欄ラベルに合算されていた（DXF-label-comparison の実データ
確認〈展開接続図23件/UNIT内結線図71件〉で、185種・870個が1行にまとまるのを発見）。
修正後: 除去すると空になる場合は元のラベルのまま比較する。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from model import assembly_compare as ac  # noqa: E402


@pytest.mark.parametrize('label', ['(BU)', '（白）', ' (BK)'])
def test_compare_key_keeps_label_when_stripping_leaves_empty(label):
    assert ac._compare_key(label) == label


def test_compare_assembly_does_not_merge_parenthesis_only_labels():
    results = {
        'a.dxf': {'title': 'UNIT内結線図', 'rows': [
            {'ラベル': '(BU)', '個数': 2}, {'ラベル': '(BK)', '個数': 1}]},
        'b.dxf': {'title': '組立図', 'rows': [{'ラベル': '(BU)', '個数': 1}]},
    }
    out = ac.compare_assembly(results, ['a.dxf'], 'b.dxf')
    labels = [r['ラベル'] for r in out['rows']]
    assert labels == ['(BK)', '(BU)']
    assert '' not in labels
