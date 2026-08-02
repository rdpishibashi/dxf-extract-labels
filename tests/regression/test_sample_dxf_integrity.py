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

カタログ同期（2026-08-02 追加）:
    `sample-dxf/README.md` は各データが「何を代表するか」を1行で記録した台帳。
    データを足したのにカタログに書かない、あるいはカタログに残したままデータを
    消す、のどちらも起きるため、両方向の一致をテストで強制する。
    後者は特に危険で、テストは `skipif(not os.path.exists(...))` で黙って緑になる
    （実際に `test_title_extraction.py` が5件中2件しか検証していなかった）。

このテストが守る不変条件:
    1. curated フォルダ間で同名ファイルが存在しないこと
       （存在すると curated 内での優先順位に依存してしまう）
    2. テストが依存する主要データが、期待するフォルダから解決されること
    3. curated フォルダの実体とカタログ（README.md）が過不足なく一致すること
"""
import os
import re
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

SAMPLE_DIR = os.path.join(PROJECT_ROOT, 'sample-dxf')
CATALOG_PATH = os.path.join(SAMPLE_DIR, 'README.md')

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


def _catalog_entries():
    """README.md のカタログ表から {ファイル名: 説明} を抽出する。

    表の行は `| \\`NAME.dxf\\` | 説明 | 参照元 |` の形式（1行1ファイル）。
    行頭が `|` かつ第1セルがバッククォート囲みの .dxf のものだけを拾うため、
    フォルダ構成表・同期テストの説明表などの他の表は自然に除外される。
    """
    entries = {}
    with open(CATALOG_PATH, encoding='utf-8') as fh:
        for line in fh:
            if not line.lstrip().startswith('|'):
                continue
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if len(cells) < 2:
                continue
            m = re.fullmatch(r'`([^`]+\.dxf)`', cells[0])
            if m:
                entries[m.group(1)] = cells[1]
    return entries


def test_catalog_covers_all_curated_files():
    """curated フォルダの全 .dxf がカタログに1行ずつ載っていること。

    載っていないデータは「なぜ必要か」が誰にも分からなくなり、増える一方で
    消せなくなる。データを追加したらカタログにも1行足すこと
    （手順は sample-dxf/README.md「データを追加するときの手順」）。
    """
    missing = sorted(set(_curated_files()) - set(_catalog_entries()))
    assert not missing, (
        'カタログ（sample-dxf/README.md）に未記載のデータがあります。'
        '「何を代表するか」を1行で追記してください:\n'
        + '\n'.join(f'  {n}' for n in missing)
    )


def test_catalog_has_no_stale_entries():
    """カタログに載っているファイルが実在すること。

    実体が消えてもテストは `skipif(not os.path.exists(...))` で黙って緑になる
    ため、欠落を検出できるのはここだけ。落ちた場合は「データを復元する」か
    「カタログから削除する」かを**意識的に**選ぶこと。
    """
    stale = sorted(set(_catalog_entries()) - set(_curated_files()))
    assert not stale, (
        'カタログに載っているが実体が無いデータがあります。'
        '復元するか、カタログから削除してください:\n'
        + '\n'.join(f'  {n}' for n in stale)
    )


def test_catalog_entries_have_description():
    """カタログの各行に説明が書かれていること（空欄・プレースホルダを禁止）。"""
    blank = sorted(
        name for name, desc in _catalog_entries().items()
        if not desc or desc in ('（要記入）', '(要記入)', '-', '—')
    )
    assert not blank, (
        'カタログの説明が未記入の行があります。用途が分からない場合は'
        '「**用途不明・要調査**」と明記してください（空欄のままにしない）:\n'
        + '\n'.join(f'  {n}' for n in blank)
    )


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
