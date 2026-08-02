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
"""
import copy
import json
import os

from model.region_detector import analyze_dxf_regions as _analyze_dxf_regions

_SAMPLE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sample-dxf'))

_region_cache = {}


def _is_shared_sample(dxf_file):
    try:
        abspath = os.path.abspath(dxf_file)
    except (TypeError, ValueError):
        return False
    return os.path.commonpath([abspath, _SAMPLE_DIR]) == _SAMPLE_DIR


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
