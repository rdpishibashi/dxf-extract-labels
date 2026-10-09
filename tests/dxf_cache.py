"""共有サンプル DXF に対する解析結果のテスト内キャッシュ。

背景（2026-08-02〜08-03）:
    `analyze_dxf_regions()` の実行時間の大半は ezdxf のパース（数秒）ではなく
    領域検出アルゴリズム自体（`EE6868-500-01C.dxf` 18.5MB で約23秒）にある。
    同じ共有サンプルに対して同じ引数で複数のテストが呼び出しており
    （`EE6868-500-01C.dxf` だけで5回）、素朴にキャッシュするだけでスイート全体の
    実行時間を大きく削減できる。詳細な計測結果は
    `HANDOFF-test-speedup.md`（削除済み・作業完了時にコミット履歴へ残す）を参照。

設計上の決めごと（変更する場合は理由を残すこと）:
    1. **`Tools/sample-dxf/` 配下のファイルだけをキャッシュする。**
       合成 DXF（`tmp_path` 配下等、テストのたびにパスが変わる）はキャッシュ対象
       にしない。対象にすると使い捨てのはずのキャッシュがテスト実行のたびに
       無制限に増える。curated フォルダの実ファイル数（50件強）で自然に頭打ちになる。
    2. **返却時に必ず `deepcopy` する。** 呼び出し側が戻り値の dict/list を書き換えても
       キャッシュ本体やレポートに再利用する結果を汚染しない。実測でコストは
       無視できる（1回23秒の解析に対し deepcopy は0.01秒未満）。
    3. **`functools.lru_cache` は使わない。** LRU 退避によって最も高価な
       `EE6868-500-01C.dxf` がテストの合間に追い出されると、削減効果がほぼ消える。
       素の dict で保持し、プロセス終了まで破棄しない。
    4. **本体コード（`model/`）には一切手を入れない。** これはテスト側だけの最適化。

追加（2026-10-09）: `extract_ref_designator_data()` / `extract_labels()` のラッパー。
    領域検出（上記）とは別に、`EE6892-455B.dxf`（40MB・読込だけで約2.8秒）や
    `EE6868-500-01C.dxf` に対して、**引数まで完全に同じ**呼び出しが複数テストに
    散らばっていた。上記1〜4をそのまま踏襲し、さらに次を守る:
    5. **キーは (関数名, ファイル内容のmd5, basename, 位置引数・キーワード引数のJSON)。**
       絶対パスではなく**内容**で引くのは、`EE6892-455B.dxf`（40MB）が中身同一のまま
       `sample-dxf/problems/` と `sample-dxf/組立図・内結線図_*/` の2か所にあり、パス
       キーだと別物扱いで重い呼び出しが重複したため（2026-10-09 実測）。結果に漏れる
       パス情報はbasenameだけなので、basenameはキーに含める。md5はパスごとに1回だけ
       計算して記憶する（40MBで約0.1秒）。ハッシュ計算に失敗した場合（読めない等）と、
       JSON化できない引数（`master_index` 等のオブジェクト）を含む場合は、
       `default=str` で無理にキー化せず**キャッシュせず素通し**する（別物が同一キーに
       なって誤ったキャッシュヒットを起こすのを防ぐ）。
    6. **キャッシュするのはこのプロジェクト自身（primary）の呼び出しだけ。** 他プロジェクト
       （DXF-diff-manager / DXF-viewer）の実装を呼ぶ側は、比較の意味が変わるため
       使わない。
"""
import copy
import hashlib
import json
import os

from model.extract_labels import extract_labels as _extract_labels
from model.ref_designator import extract_ref_designator_data as _extract_ref_designator_data
from model.region_detector import analyze_dxf_regions as _analyze_dxf_regions

_SAMPLE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sample-dxf'))

_region_cache = {}
_call_cache = {}
_digest_memo = {}


def _is_shared_sample(dxf_file):
    """`sample-dxf/` 配下のファイルか。シンボリックリンクを解決した実パスで比べる。

    プロジェクト直下の `sample-dxf` は `../sample-dxf` へのsymlinkで、テストによって
    `<project>/sample-dxf/...`（リンク経由）と `<Tools>/sample-dxf/...`（実パス）の
    どちらでもファイルを指す。文字列のままだと後者が「共有サンプルではない」と
    判定されてキャッシュされず、組立図テストなどの重い呼び出しが素通しになっていた
    （2026-10-09 実測）。"""
    try:
        real = os.path.realpath(dxf_file)
        sample_real = os.path.realpath(_SAMPLE_DIR)
        return os.path.commonpath([real, sample_real]) == sample_real
    except (TypeError, ValueError):
        return False


def _cache_key(dxf_file, config):
    return (os.path.abspath(dxf_file), json.dumps(config, sort_keys=True, default=str))


def analyze_dxf_regions(dxf_file, config=None):
    """`model.region_detector.analyze_dxf_regions` のキャッシュ付きラッパー。

    共有サンプル（sample-dxf/ 配下）以外はキャッシュせず素通しする。
    """
    if not _is_shared_sample(dxf_file):
        return _analyze_dxf_regions(dxf_file, config)

    key = _cache_key(dxf_file, config)
    if key not in _region_cache:
        _region_cache[key] = _analyze_dxf_regions(dxf_file, config)
    return copy.deepcopy(_region_cache[key])


def _file_digest(dxf_file):
    """ファイル内容のmd5（絶対パスごとに1回だけ計算して記憶する）。

    共有サンプルはテスト実行中に変更されない前提（`sample-dxf/` は固定データ）。"""
    abspath = os.path.abspath(dxf_file)
    if abspath not in _digest_memo:
        h = hashlib.md5()
        with open(abspath, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        _digest_memo[abspath] = h.hexdigest()
    return _digest_memo[abspath]


def _cached_call(name, func, dxf_file, args, kwargs):
    """`func(dxf_file, *args, **kwargs)` の結果を共有サンプルに限りキャッシュする。

    共有サンプル以外、ハッシュ計算に失敗した場合、JSON化できない引数を含む場合は、
    キャッシュせず素通しする。
    """
    if not _is_shared_sample(dxf_file):
        return func(dxf_file, *args, **kwargs)
    try:
        key = (name, _file_digest(dxf_file), os.path.basename(dxf_file),
               json.dumps([list(args), kwargs], sort_keys=True))
    except (TypeError, OSError):
        return func(dxf_file, *args, **kwargs)
    if key not in _call_cache:
        _call_cache[key] = func(dxf_file, *args, **kwargs)
    return copy.deepcopy(_call_cache[key])


def extract_ref_designator_data(dxf_file, *args, **kwargs):
    """`model.ref_designator.extract_ref_designator_data` のキャッシュ付きラッパー。"""
    return _cached_call('extract_ref_designator_data', _extract_ref_designator_data,
                        dxf_file, args, kwargs)


def extract_labels(dxf_file, *args, **kwargs):
    """`model.extract_labels.extract_labels` のキャッシュ付きラッパー（戻り値は (labels, info)）。"""
    return _cached_call('extract_labels', _extract_labels, dxf_file, args, kwargs)
