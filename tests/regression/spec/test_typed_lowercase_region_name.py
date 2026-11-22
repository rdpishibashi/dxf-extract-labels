"""仕様: 名称のない領域は、「領域名を入力」でタイプインした小文字のラベルを名称として採用する（自動では名称にしない）。

**受入条件**（2026-11-22、`EE6888-604-01A.dxf` の `Shield box`〈5.1%〉・`Wafer stage`〈1.8%〉。ユーザー決定:
primary では小文字のラベルを自動の名称にしないが、「領域名を入力」でタイプインした場合は採用する）:

- Given 既定の設定、When `analyze_dxf_regions()`、Then 小文字を含むラベル（`Shield box`・`Wafer stage`）は名称候補にならず、
  その領域（477.1〜552.1 × 225.1〜370.5 の 5.1%）は名称なし（`default_name` が空）。
  （`name_exclude_lowercase` は True のまま。False にすると 58 ファイルが変わり、`W No.…`・`PIN No.` のような配線・端子の注記が名称になって悪化する。
  名称なしの領域に限って自動で小文字を許す案も、273 ファイルで 5 領域〈`PIN No.` を含む〉が変わるため採用しない。）
- Given 「領域名を入力」に `shield BOX`（大文字小文字・全角半角を問わない）、Then 採用される名称は図面の元の文字列 `Shield box`
  （`resolve_name_selections()` → `build_named_regions()` の領域名）。
- Given 一致しない入力（`Shield`）、Then 未選択（`['']`）。

境界・例外の意図: 入力の照合は `region_choices.match_typed_region_name()`（NFKC・大文字化の完全一致）で、照合対象は枠内の**全ラベル**
（`analysis['labels']`。領域名の候補のフィルタ〈小文字除外など〉を通らない）。そのため追加のコードは要らず、このテストが仕様として固定する。
Cable-checker は独自の `gapped_frame_regions()` でこの外周を領域にしているので、小文字のオプションは使わない。
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tests'))

from model import ref_designator, region_choices as rc  # noqa: E402
from model.region_decisions import classify_regions  # noqa: E402
from model.region_detector import DEFAULT_REGION_CONFIG, analyze_dxf_regions  # noqa: E402
from sample_data import require_sample  # noqa: E402

pytestmark = pytest.mark.slow

SAMPLE = 'EE6888-604-01A.dxf'
APP_CFG = {'detect_minor_regions': True, 'local_rotation': True}
BBOX = (477.1, 552.1, 225.1, 370.5)      # Shield box ラベルがある、名称なしの 5.1% の領域


@pytest.fixture(scope='module')
def analysis():
    path = require_sample(SAMPLE)
    if path is None:
        pytest.skip('sample-dxf が無い環境')
    result = analyze_dxf_regions(path, APP_CFG)
    assert result['error'] is None
    return result


def _unnamed_region(analysis):
    for r in analysis['regions']:
        xs = [p[0] for p in r['polygon']]
        ys = [p[1] for p in r['polygon']]
        got = (min(xs), max(xs), min(ys), max(ys))
        if all(abs(a - b) <= 0.1 for a, b in zip(got, BBOX)):
            return r
    raise AssertionError('Shield box の領域（477.1〜552.1 × 225.1〜370.5）が検出されない')


def _selection(analysis, typed_text):
    state = {}
    region = _unnamed_region(analysis)
    rc.set_value(state, SAMPLE, region['id'], 'typed', True)
    rc.set_value(state, SAMPLE, region['id'], 'typedtext', typed_text)
    names = {c[1] for r in analysis['regions'] for c in r['name_candidates']}
    decisions = classify_regions(analysis, names, set(), lambda _label: True)
    selections = rc.resolve_name_selections(state, SAMPLE, analysis, decisions)
    return region, selections


def test_lowercase_labels_are_not_automatic_names(analysis):
    assert DEFAULT_REGION_CONFIG['name_exclude_lowercase'] is True
    region = _unnamed_region(analysis)
    assert region['default_name'] == '' and region['name_candidates'] == []
    assert round(region['area_pct'], 1) == 5.1


@pytest.mark.parametrize('typed', ['Shield box', 'shield BOX', 'ＳＨＩＥＬＤ　ＢＯＸ'])
def test_a_typed_lowercase_label_is_adopted_as_the_region_name(analysis, typed):
    region, selections = _selection(analysis, typed)
    assert selections[(SAMPLE, region['id'])] == ['Shield box']          # 図面の元の文字列
    named, _next_index = ref_designator.build_named_regions(analysis, selections, SAMPLE)
    assert ('Shield box', 5.1) in [(n['name'], round(n['area_pct'], 1)) for n in named]


def test_a_typed_name_that_matches_no_label_is_not_adopted(analysis):
    region, selections = _selection(analysis, 'Shield')
    assert selections[(SAMPLE, region['id'])] == ['']


def test_other_lowercase_label_matches_too(analysis):
    assert rc.match_typed_region_name('wafer stage', analysis['labels']) == 'Wafer stage'
