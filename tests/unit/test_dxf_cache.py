"""tests/dxf_cache.py（共有サンプルDXFの解析結果キャッシュ）の単体テスト。

背景: `analyze_dxf_regions()` は実行コストの大半が領域検出アルゴリズムに
あり（`HANDOFF-test-speedup.md` 参照）、同じ共有サンプルへの重複呼び出しを
キャッシュで排除する。キャッシュ自体が壊れると「別内容のはずの解析結果を
使い回す」「1つのテストの変更が他のテストに伝播する」といった検出困難な
不具合を生むため、ここで不変条件を固定する。
"""
import os
import sys
import tempfile

import ezdxf
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from tests import dxf_cache  # noqa: E402

SAMPLE_DIR = os.path.join(PROJECT_ROOT, 'sample-dxf')
SHARED_SAMPLE = os.path.join(SAMPLE_DIR, 'samples', 'EE6097-039-06C.dxf')  # 3図面枠・軽量


@pytest.fixture(autouse=True)
def _clear_cache():
    """テスト間でキャッシュ状態が漏れないようにする（他テストの実行順に非依存）。"""
    dxf_cache._region_cache.clear()
    yield
    dxf_cache._region_cache.clear()


def _make_synthetic_doc():
    doc = ezdxf.new()
    msp = doc.modelspace()
    for p1, p2 in [((0, 0), (200, 0)), ((200, 0), (200, 100)),
                   ((200, 100), (0, 100)), ((0, 100), (0, 0))]:
        msp.add_line(p1, p2, dxfattribs={'lineweight': 100, 'color': 7})
    return doc


@pytest.mark.skipif(not os.path.exists(SHARED_SAMPLE), reason='サンプル DXF が無い')
def test_second_call_returns_equivalent_result():
    """同一引数の2回目呼び出しが1回目と等価な結果を返す（キャッシュヒット時も内容は同じ）。"""
    a1 = dxf_cache.analyze_dxf_regions(SHARED_SAMPLE)
    a2 = dxf_cache.analyze_dxf_regions(SHARED_SAMPLE)
    assert a1 == a2


@pytest.mark.skipif(not os.path.exists(SHARED_SAMPLE), reason='サンプル DXF が無い')
def test_second_call_returns_distinct_object():
    """2回目の戻り値は1回目とは別オブジェクト（deepcopy による分離を検証）。"""
    a1 = dxf_cache.analyze_dxf_regions(SHARED_SAMPLE)
    a2 = dxf_cache.analyze_dxf_regions(SHARED_SAMPLE)
    assert a1 is not a2
    assert a1['regions'] is not a2['regions']


@pytest.mark.skipif(not os.path.exists(SHARED_SAMPLE), reason='サンプル DXF が無い')
def test_mutating_returned_result_does_not_pollute_cache():
    """呼び出し側が戻り値を書き換えても、以降の呼び出し結果に影響しない（変更汚染の隔離）。"""
    a1 = dxf_cache.analyze_dxf_regions(SHARED_SAMPLE)
    a1['regions'].clear()
    a1['frames'] = 'CORRUPTED'

    a2 = dxf_cache.analyze_dxf_regions(SHARED_SAMPLE)
    assert a2['regions'] != []
    assert a2['frames'] != 'CORRUPTED'


@pytest.mark.skipif(not os.path.exists(SHARED_SAMPLE), reason='サンプル DXF が無い')
def test_different_config_produces_separate_cache_entries():
    """config が異なれば別キーとして扱われ、結果が混同されない（キー衝突が無いこと）。"""
    a_default = dxf_cache.analyze_dxf_regions(SHARED_SAMPLE, None)
    a_custom = dxf_cache.analyze_dxf_regions(SHARED_SAMPLE, {'area_ratio': 0.5})
    assert len(dxf_cache._region_cache) == 2
    # area_ratio を極端に上げると採用される領域が減る（または変わる）はず。
    assert a_default != a_custom or a_default['regions'] == a_custom['regions'] == []


def test_synthetic_path_outside_sample_dir_is_not_cached():
    """sample-dxf/ 配下ではない一時ファイルはキャッシュされない（素通し）。"""
    doc = _make_synthetic_doc()
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as f:
        path = f.name
    doc.saveas(path)
    try:
        dxf_cache.analyze_dxf_regions(path)
        assert len(dxf_cache._region_cache) == 0
    finally:
        os.remove(path)
