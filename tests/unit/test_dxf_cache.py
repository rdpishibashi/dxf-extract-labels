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
    for c in (dxf_cache._region_cache, dxf_cache._call_cache, dxf_cache._digest_memo):
        c.clear()
    yield
    for c in (dxf_cache._region_cache, dxf_cache._call_cache, dxf_cache._digest_memo):
        c.clear()


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


# ---------------------------------------------------------------------------
# extract_ref_designator_data / extract_labels ラッパー（2026-10-09）
#
# 実DXFは解析せず、呼び出し回数を数える偽関数に差し替えて、キャッシュ自体の
# 不変条件（ヒット・汚染隔離・キー分離・素通し）だけを高速に検証する。
# ---------------------------------------------------------------------------

# (ラッパー名, 差し替える内部名, 偽の戻り値を作る関数)
_WRAPPERS = [
    ('extract_ref_designator_data', '_extract_ref_designator_data',
     lambda: {'labels': [('CB001', 1.0, 2.0)], 'rows': [{'ラベル': 'CB001'}]}),
    ('extract_labels', '_extract_labels',
     lambda: (['R1', 'R2'], {'title': 'T', 'subtitle': ['S']})),
]
_SHARED_PATH = os.path.join(SAMPLE_DIR, 'samples', 'not-read-by-fake.dxf')


@pytest.fixture
def digests(monkeypatch):
    """`_file_digest` を差し替える（実ファイルを読まない）。

    戻り値の dict に {絶対パス: digest} を入れると、そのパスの内容ハッシュとして扱われる。
    未登録のパスは basename ごとに固有のdigestになる。"""
    table = {}
    monkeypatch.setattr(
        dxf_cache, '_file_digest',
        lambda p: table.get(os.path.abspath(p), 'digest-of-' + os.path.basename(p)))
    return table


@pytest.fixture
def counting_fake(monkeypatch, digests):
    """内部関数を差し替え、呼び出し回数を数える。実ファイルは読まれない。"""
    def install(inner_name, make_result):
        calls = []

        def fake(dxf_file, *args, **kwargs):
            calls.append((dxf_file, args, kwargs))
            return make_result()

        monkeypatch.setattr(dxf_cache, inner_name, fake)
        return calls
    return install


@pytest.mark.parametrize('wrapper_name, inner_name, make_result', _WRAPPERS)
def test_call_wrapper_hits_cache_for_identical_arguments(
        counting_fake, wrapper_name, inner_name, make_result):
    """引数まで同一の2回目はキャッシュヒットし、内部関数は1回しか呼ばれない。"""
    calls = counting_fake(inner_name, make_result)
    wrapper = getattr(dxf_cache, wrapper_name)
    r1 = wrapper(_SHARED_PATH, frame_lineweight=100, original_filename='a.dxf')
    r2 = wrapper(_SHARED_PATH, frame_lineweight=100, original_filename='a.dxf')
    assert len(calls) == 1
    assert r1 == r2


@pytest.mark.parametrize('wrapper_name, inner_name, make_result', _WRAPPERS)
def test_call_wrapper_isolates_mutation_of_returned_value(
        counting_fake, wrapper_name, inner_name, make_result):
    """戻り値（タプル内のlist/dictも含む）を書き換えても、次の呼び出しに伝播しない。"""
    counting_fake(inner_name, make_result)
    wrapper = getattr(dxf_cache, wrapper_name)
    r1 = wrapper(_SHARED_PATH, original_filename='a.dxf')
    expected = make_result()
    first_container = r1[0] if isinstance(r1, tuple) else r1['labels']
    first_container.append('CORRUPTED')
    if isinstance(r1, tuple):
        r1[1]['title'] = 'CORRUPTED'
    else:
        r1['rows'].clear()
    r2 = wrapper(_SHARED_PATH, original_filename='a.dxf')
    assert r2 == expected
    assert r2 is not r1


@pytest.mark.parametrize('wrapper_name, inner_name, make_result', _WRAPPERS)
def test_call_wrapper_separates_entries_by_arguments_and_function(
        counting_fake, wrapper_name, inner_name, make_result):
    """引数が1つでも違えば別キー。同じパスでも別関数のキャッシュとは混ざらない。"""
    calls = counting_fake(inner_name, make_result)
    wrapper = getattr(dxf_cache, wrapper_name)
    wrapper(_SHARED_PATH, original_filename='a.dxf')
    wrapper(_SHARED_PATH, original_filename='b.dxf')
    wrapper(_SHARED_PATH, 'positional', original_filename='a.dxf')
    assert len(calls) == 3
    assert len(dxf_cache._call_cache) == 3
    assert {k[0] for k in dxf_cache._call_cache} == {wrapper_name}


@pytest.mark.parametrize('wrapper_name, inner_name, make_result', _WRAPPERS)
def test_call_wrapper_passes_through_non_json_arguments(
        counting_fake, wrapper_name, inner_name, make_result):
    """JSON化できない引数（master_index 等のオブジェクト）を含む呼び出しはキャッシュしない。

    str() で無理にキー化すると別オブジェクトが同一キーになり得るため、素通しにする。"""
    calls = counting_fake(inner_name, make_result)
    wrapper = getattr(dxf_cache, wrapper_name)
    wrapper(_SHARED_PATH, master_index=object())
    wrapper(_SHARED_PATH, master_index=object())
    assert len(calls) == 2
    assert dxf_cache._call_cache == {}


@pytest.mark.parametrize('wrapper_name, inner_name, make_result', _WRAPPERS)
def test_call_wrapper_does_not_cache_paths_outside_sample_dir(
        counting_fake, tmp_path, wrapper_name, inner_name, make_result):
    """sample-dxf/ 配下ではないパス（合成DXF・tmp_path）は素通しで、キャッシュが増えない。"""
    calls = counting_fake(inner_name, make_result)
    wrapper = getattr(dxf_cache, wrapper_name)
    path = str(tmp_path / 'synthetic.dxf')
    wrapper(path, original_filename='a.dxf')
    wrapper(path, original_filename='a.dxf')
    assert len(calls) == 2
    assert dxf_cache._call_cache == {}


@pytest.mark.parametrize('wrapper_name, inner_name, make_result', _WRAPPERS)
def test_call_wrapper_shares_entries_between_identical_content_at_different_paths(
        counting_fake, digests, wrapper_name, inner_name, make_result):
    """中身が同一（digest一致）でbasenameも同じなら、別ディレクトリのコピーでもヒットする。

    `EE6892-455B.dxf` が `problems/` と `組立図・内結線図_*/` に同一内容で存在し、
    パスをキーにすると重い呼び出しが重複していた（2026-10-09 実測）ことの再発防止。"""
    calls = counting_fake(inner_name, make_result)
    wrapper = getattr(dxf_cache, wrapper_name)
    p1 = os.path.join(SAMPLE_DIR, 'problems', 'same.dxf')
    p2 = os.path.join(SAMPLE_DIR, 'other-dir', 'same.dxf')
    digests[os.path.abspath(p1)] = digests[os.path.abspath(p2)] = 'D'
    wrapper(p1, original_filename='same.dxf')
    wrapper(p2, original_filename='same.dxf')
    assert len(calls) == 1


@pytest.mark.parametrize('wrapper_name, inner_name, make_result', _WRAPPERS)
def test_call_wrapper_separates_entries_by_content_or_basename(
        counting_fake, digests, wrapper_name, inner_name, make_result):
    """同一パス構成でも、内容(digest)が違う、またはbasenameが違えば別キー。"""
    calls = counting_fake(inner_name, make_result)
    wrapper = getattr(dxf_cache, wrapper_name)
    a = os.path.join(SAMPLE_DIR, 'problems', 'a.dxf')
    b = os.path.join(SAMPLE_DIR, 'other-dir', 'a.dxf')       # 同basename・別内容
    c = os.path.join(SAMPLE_DIR, 'other-dir', 'c.dxf')       # 同内容・別basename
    digests[os.path.abspath(a)] = 'D1'
    digests[os.path.abspath(b)] = 'D2'
    digests[os.path.abspath(c)] = 'D1'
    for path in (a, b, c):
        wrapper(path)
    assert len(calls) == 3


@pytest.mark.parametrize('wrapper_name, inner_name, make_result', _WRAPPERS)
def test_call_wrapper_passes_through_when_digest_fails(
        monkeypatch, counting_fake, wrapper_name, inner_name, make_result):
    """ハッシュ計算に失敗（読めない等）したらキャッシュせず素通しする。"""
    calls = counting_fake(inner_name, make_result)

    def boom(_p):
        raise OSError('unreadable')
    monkeypatch.setattr(dxf_cache, '_file_digest', boom)
    wrapper = getattr(dxf_cache, wrapper_name)
    wrapper(_SHARED_PATH)
    wrapper(_SHARED_PATH)
    assert len(calls) == 2
    assert dxf_cache._call_cache == {}


def test_file_digest_depends_on_content_and_is_memoized(tmp_path):
    """_file_digest は内容で決まり（パス非依存）、同じパスの2回目は再計算しない。"""
    f1, f2, f3 = (tmp_path / n for n in ('a.bin', 'b.bin', 'c.bin'))
    f1.write_bytes(b'same'); f2.write_bytes(b'same'); f3.write_bytes(b'diff')
    assert dxf_cache._file_digest(str(f1)) == dxf_cache._file_digest(str(f2))
    assert dxf_cache._file_digest(str(f1)) != dxf_cache._file_digest(str(f3))
    f1.write_bytes(b'changed-after-first-call')
    assert dxf_cache._file_digest(str(f1)) == dxf_cache._file_digest(str(f2))  # 記憶済み


def test_is_shared_sample_resolves_symlinks(tmp_path, monkeypatch):
    """symlink経由でも実パス経由でも、sample-dxf 配下なら共有サンプルと判定する。

    プロジェクト直下の `sample-dxf` は `../sample-dxf` へのsymlinkで、テストによって
    リンク経由・実パス経由のどちらでも同じファイルを指す。文字列比較のままだと
    実パス側が素通しになりキャッシュが効かなかった（2026-10-09）ことの再発防止。"""
    real_dir = tmp_path / 'Tools' / 'sample-dxf'
    real_dir.mkdir(parents=True)
    (real_dir / 'a.dxf').write_bytes(b'x')
    project = tmp_path / 'Tools' / 'proj'
    project.mkdir()
    link = project / 'sample-dxf'
    link.symlink_to(real_dir, target_is_directory=True)
    outside = tmp_path / 'Tools' / 'elsewhere.dxf'
    outside.write_bytes(b'x')

    monkeypatch.setattr(dxf_cache, '_SAMPLE_DIR', str(link))  # <project>/sample-dxf（リンク）
    assert dxf_cache._is_shared_sample(str(link / 'a.dxf'))         # リンク経由
    assert dxf_cache._is_shared_sample(str(real_dir / 'a.dxf'))     # 実パス経由
    assert not dxf_cache._is_shared_sample(str(outside))
