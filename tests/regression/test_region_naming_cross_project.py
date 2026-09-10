"""DXF-viewer（`core/region_detector.py`）との矩形領域検出クロスプロジェクト一貫性テスト。

背景（2026-07-16）: DXF-extract-labelsでは最小領域サイズ3%で"CM DRV"領域を抽出
できるのに、DXF-viewerでは1%にしても認識できないという不具合が発生した。原因は
DXF-viewer側の領域名候補フィルタ3箇所で`normalize_width()`の適用が漏れており、
全角の機器符号・除外語が正規化なしのASCII前提パターンに素通りしていたこと
（2026-07-03の全角/半角対応が`_count_letters`にしか移植されておらず、機器符号・
除外語マッチの正規化漏れには誰も気づいていなかった）。

`region_detector.py`は`extract_labels.py`と異なりバイト一致コピーの維持対象では
なく（プロジェクトごとに依存関数を自己完結化する「移植」方針、詳細は
`~/.claude/skills/dxf-electrical-circuit/references/region-detection.md`
10節相当）、テキストdiffだけでは今回のような「アルゴリズムは同じに見えるが
一部の判定ロジックだけ移植漏れ」という不具合を検出できない。そのため実データ
（`sample-dxf/problems/`）で両プロジェクトの検出結果そのもの（領域名の集合）を
突き合わせる、振る舞いベースの一貫性テストをここに新設する。

**region_detector.py を変更したら（どちらのプロジェクト側でも）このテストを
実行し、両プロジェクトの検出結果が一致することを確認すること。**

DXF-viewer が見つからない環境（DXF-extract-labels 単独チェックアウト等）では
全テストをスキップする。
"""
import glob
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model.region_detector import (  # noqa: E402
    analyze_dxf_regions as analyze_primary,
    DEFAULT_REGION_CONFIG as PRIMARY_CONFIG,
)

TOOLS_ROOT = os.path.dirname(PROJECT_ROOT)
DXF_VIEWER_ROOT = os.path.join(TOOLS_ROOT, 'DXF-viewer')
SAMPLE_PROBLEMS_DIR = os.path.join(TOOLS_ROOT, 'sample-dxf', 'problems')

_viewer_available = os.path.isfile(os.path.join(DXF_VIEWER_ROOT, 'core', 'region_detector.py'))
if _viewer_available:
    sys.path.insert(0, DXF_VIEWER_ROOT)
    from core.region_detector import (  # noqa: E402
        analyze_dxf_regions as analyze_viewer,
        DEFAULT_REGION_CONFIG as VIEWER_CONFIG,
    )

pytestmark = pytest.mark.skipif(
    not _viewer_available,
    reason="DXF-viewer が見つからないためクロスプロジェクトテストをスキップ（サブフォルダとして同じTools/配下にチェックアウトされている環境でのみ実行）",
)

# DEFAULT_REGION_CONFIG の差異のうち、意図的な機能差として許容するキー。
# それ以外の値差は primary からの意図しないドリフトとみなし失敗させる。
# name_filter_prefixes: DXF-extract-labels のみが持つUI機能（v1.9.3）。DXF-viewer に同機能はない。
ALLOWED_CONFIG_ONLY_IN_PRIMARY = {'name_filter_prefixes'}


def _sample_problem_files():
    if not os.path.isdir(SAMPLE_PROBLEMS_DIR):
        return []
    return sorted(glob.glob(os.path.join(SAMPLE_PROBLEMS_DIR, '*.dxf')))


def test_default_region_config_parity():
    """DEFAULT_REGION_CONFIG の値が両プロジェクトで一致すること（意図的な機能差を除く）。

    2026-07-16 時点で `connection_point_margin` が primary=0.05 / viewer=0.1 と
    気づかれずにドリフトしていたことが発覚した（本テスト新設のきっかけ）。
    """
    primary_only = set(PRIMARY_CONFIG) - set(VIEWER_CONFIG)
    assert primary_only == ALLOWED_CONFIG_ONLY_IN_PRIMARY, (
        f"primaryにのみ存在するキーが想定外です: {primary_only - ALLOWED_CONFIG_ONLY_IN_PRIMARY}"
    )
    viewer_only = set(VIEWER_CONFIG) - set(PRIMARY_CONFIG)
    assert not viewer_only, f"DXF-viewerにのみ存在するキーがあります: {viewer_only}"

    mismatches = []
    for key in sorted(set(PRIMARY_CONFIG) & set(VIEWER_CONFIG)):
        if PRIMARY_CONFIG[key] != VIEWER_CONFIG[key]:
            mismatches.append((key, PRIMARY_CONFIG[key], VIEWER_CONFIG[key]))
    assert not mismatches, (
        "DEFAULT_REGION_CONFIG の値がドリフトしています（primary値, viewer値）: "
        f"{mismatches}"
    )


@pytest.mark.parametrize("dxf_path", _sample_problem_files(), ids=lambda p: os.path.basename(p))
def test_region_naming_matches_across_projects(dxf_path):
    """同一の設定（primaryのDEFAULT_REGION_CONFIG）を明示的に渡した上で、
    両プロジェクトが検出する領域（面積・名称）の集合が完全一致すること。

    設定を明示的に固定して渡すため、両プロジェクトのDEFAULT_REGION_CONFIG自体が
    将来ドリフトしても本テストは「アルゴリズムの一致」だけを純粋に検証する
    （設定ドリフト自体は test_default_region_config_parity が別途検出する）。
    """
    cfg = dict(PRIMARY_CONFIG)
    cfg.pop('name_filter_prefixes', None)  # DXF-viewer側のconfigに無いキーは渡さない

    a_primary = analyze_primary(dxf_path, cfg)
    a_viewer = analyze_viewer(dxf_path, cfg)

    # 図面枠が Paper Space にしか無い等、意図的にエラーになるファイルもある
    # （例: EE5322-455-01B.dxf）。その場合も両プロジェクトが同じエラーメッセージ
    # で失敗することを求める（メッセージ文言のドリフトも実際に発生した実績あり、
    # 2026-07-16 に本テスト新設時 primary の v1.8.3 文言変更が viewer 未移植と判明）。
    assert a_primary.get('error') == a_viewer.get('error'), (
        f"{os.path.basename(dxf_path)}: エラーの有無/文言が一致しません。\n"
        f"primary: {a_primary.get('error')!r}\n"
        f"viewer:  {a_viewer.get('error')!r}"
    )
    if a_primary.get('error') is not None:
        return

    def _signature(analysis):
        return sorted(
            (round(r['area'], 1), r.get('default_name') or '')
            for r in analysis['regions']
        )

    sig_primary = _signature(a_primary)
    sig_viewer = _signature(a_viewer)
    assert sig_primary == sig_viewer, (
        f"{os.path.basename(dxf_path)}: 領域の(面積, 名称)集合がprimaryとDXF-viewerで異なります。\n"
        f"primaryのみ: {sorted(set(sig_primary) - set(sig_viewer))}\n"
        f"viewerのみ: {sorted(set(sig_viewer) - set(sig_primary))}"
    )
