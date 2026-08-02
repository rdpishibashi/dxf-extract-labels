"""共有テストデータ（Tools/sample-dxf）の健全性を守る回帰テスト。

背景（2026-08-02）:
    `_find_sample(name)` は sample-dxf 配下をファイル名で再帰検索する。
    この設計自体は「ファイルを別サブフォルダへ移動してもテストが壊れない」
    という利点があるが、**同名ファイルが複数存在すると `os.walk()` の到達順で
    どちらを掴むかが決まる**ため、テストが意図しない版のDXFを読む危険があった。

    実際に、テストが `_find_sample()` でロードしている9ファイルが同名重複して
    おり、うち6件は内容も異なっていた。特に `EE6868-500-01C.dxf`（13図面枠の
    基準データ、8テストが使用し `len(frames) == 13` を検証）は
    `samples/` と `pairC/…（比較元）/` に別内容で存在し、走査順が変われば
    テスト群が一斉に落ちる（または誤ったデータで通る）状態だった。

    なお pairC は「同一図番の比較元/比較先」を並べたペア比較用データであり、
    **同名・別内容であること自体が設計上正しい**。したがってデータ側を改名
    するのではなく、`_find_sample()` が curated フォルダ
    （problems / samples / terminal-detector / viewer-error）を固定順で
    先に探すようにして解決した。

このテストが守る不変条件:
    1. curated フォルダ間で同名ファイルが存在しないこと
       （存在すると curated 内での優先順位に依存してしまう）
    2. テストが依存する主要データが、期待するフォルダから解決されること
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

SAMPLE_DIR = os.path.join(PROJECT_ROOT, 'sample-dxf')

# `_find_sample()` が固定順で探索する curated フォルダ（実装と同じ順序・同じ集合）。
# pairC は同名・別内容が設計上正当なため、意図的に含めない。
CURATED_SUBDIRS = ('problems', 'samples', 'terminal-detector', 'viewer-error')

pytestmark = pytest.mark.skipif(
    not os.path.isdir(SAMPLE_DIR), reason='共有 sample-dxf が無い環境')


def _curated_files():
    """curated フォルダ直下の .dxf を {ファイル名: [パス, ...]} で返す。"""
    found = {}
    for sub in CURATED_SUBDIRS:
        d = os.path.join(SAMPLE_DIR, sub)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith('.dxf'):
                found.setdefault(f, []).append(os.path.join(sub, f))
    return found


def test_no_duplicate_filenames_across_curated_subdirs():
    """curated フォルダ間で同名ファイルが重複しないこと。

    重複すると `_find_sample()` の戻り値が CURATED_SUBDIRS の並び順という
    実装詳細に依存してしまい、フォルダを整理しただけでテストの対象データが
    変わりうる。重複を見つけたら、用途に合う1箇所だけを残すこと。
    """
    dupes = {name: paths for name, paths in _curated_files().items() if len(paths) > 1}
    assert not dupes, (
        'curated フォルダ間で同名ファイルが重複しています。'
        'どちらが使われるかが探索順に依存するため、1箇所に統合してください:\n'
        + '\n'.join(f'  {n}: {p}' for n, p in sorted(dupes.items()))
    )


@pytest.mark.parametrize('name, expected_subdir', [
    # 13図面枠の基準データ。pairC にも同名・別内容が存在する
    ('EE6868-500-01C.dxf', 'samples'),
    # viewer-error 由来。pairC にも同名・別内容が2つ存在する
    ('EE6313-545-01D.dxf', 'viewer-error'),
    # 以下は元々 pairC にしか無く、不具合再現データとして problems/ へ集約した
    ('DE5027-558-01B.dxf', 'problems'),   # SY12 RX（Tier2が正解）ケース
    ('DE5027-563-03A.dxf', 'problems'),   # IOG B7（頻度は別フレーム数）ケース
    ('DE5327-405-26A.dxf', 'problems'),   # CTC（unit_names 誤除外）ケース
    ('EE6868-516-01A.dxf', 'problems'),   # 高さの異なる2枠が左辺を共有するケース
])
def test_key_samples_resolve_to_expected_subdir(name, expected_subdir):
    """主要データが期待する curated フォルダから解決されること。

    いずれも sample-dxf 内に同名ファイルが複数存在するため、解決先が変わると
    テストが別内容のDXFを読むことになる（アサーション値は解決先の内容に
    依存している）。
    """
    from tests.regression.test_region_extraction import _find_sample

    path = _find_sample(name)
    assert path and os.path.exists(path), f'{name} が見つかりません'
    actual = os.path.basename(os.path.dirname(path))
    assert actual == expected_subdir, (
        f'{name} が {expected_subdir}/ ではなく {actual}/ から解決されました。'
        'テストが別内容のDXFを読んでいる可能性があります。'
    )
