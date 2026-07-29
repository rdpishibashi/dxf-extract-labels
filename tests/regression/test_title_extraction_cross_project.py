"""DXF-diff-manager / DXF-visual-diff との Title/Subtitle抽出クロスプロジェクト一貫性テスト。

背景（2026-07-29）: `model/extract_labels.py` 自体はプロジェクト間でbyte一致していても、
呼び出し側が渡すオプションの組み合わせが食い違うと結果が食い違う、という不具合クラスが
判明した。`extract_title_option=True` と `extract_drawing_numbers_option=True` は
セットで渡さないと、`main_drawing_group` によるタイトルブロックグループ制限
（v1.7.12）が働かず、複数タイトルブロックが1ファイルに存在する図面（同一図番の
シートが横並びで、対象シートだけタイトル行が欠落している等）でTitle/Subtitleを
誤抽出する。`DXF-diff-manager/model/label_diff.py` の呼び出し漏れで実際に発生し、
2026-07-29に修正済み（詳細は `sample-dxf/problems/DE3527-553-10E.dxf`）。

`extract_labels.py` 自体のdiff/md5確認だけではこの種の不具合（呼び出し側の乖離）を
検知できないため、実データで各プロジェクトの「実際の呼び出しパス」
（`compute_label_differences()`）を通した結果を、primary の正しい呼び出し
（`extract_labels(extract_drawing_numbers_option=True, extract_title_option=True, ...)`）
と突き合わせる、振る舞いベースの一貫性テストをここに新設する
（`region_detector.py` の `test_region_naming_cross_project.py` と同じ枠組み）。

**`extract_labels()` の呼び出しに関わる変更（呼び出し元も含む）を行ったら、この
テストを実行して結果が一致することを確認すること。**

DXF-diff-manager / DXF-visual-diff が見つからない環境ではそれぞれスキップする。

追記（2026-07-29、同日中）: 本テスト新設時点では `DXF-visual-diff/utils/label_diff.py` が
`extract_drawing_numbers_option=True` を渡しておらず7件で誤抽出することを確認し
`xfail` にしていたが、その後 primary 側で根本対応（v1.9.22、`extract_title_option=True`
であれば内部的にタイトルブロックグループ判定を行う設計変更）を行い `extract_labels.py`
を同期した結果、`label_diff.py` 自体は無変更のままこの不具合が副次的に解消された
（`xfail` が `XPASS(strict)` に転じたため撤去）。**共有ファイルの根本対応は、呼び出し側を
直接修正していないプロジェクトにも波及し得る**、という実例として記録しておく。
"""
import glob
import importlib
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model.extract_labels import extract_labels  # noqa: E402

TOOLS_ROOT = os.path.dirname(PROJECT_ROOT)
DIFF_MANAGER_ROOT = os.path.join(TOOLS_ROOT, 'DXF-diff-manager')
VISUAL_DIFF_ROOT = os.path.join(TOOLS_ROOT, 'DXF-visual-diff')
SAMPLE_PROBLEMS_DIR = os.path.join(TOOLS_ROOT, 'sample-dxf', 'problems')

_diff_manager_available = os.path.isfile(os.path.join(DIFF_MANAGER_ROOT, 'model', 'label_diff.py'))
_visual_diff_available = os.path.isfile(os.path.join(VISUAL_DIFF_ROOT, 'utils', 'label_diff.py'))

def _import_foreign_module(project_root, top_level_package, module_name):
    """他プロジェクトのモジュールを、パッケージ名衝突から隔離してインポートする。

    `DXF-diff-manager` は最上位パッケージ名が primary（本プロジェクト）自身と同じ
    `model` で衝突する。`DXF-visual-diff` は `utils` を使うため一見衝突しないが、
    `DXF-viewer/core/region_detector.py` が自身の兄弟パッケージ `utils` を
    `from utils.text_utils import ...` で参照しており、`test_region_naming_cross_project.py`
    経由でDXF-viewerがsys.pathに追加された状態でテストスイート全体を実行すると、
    `sys.modules['utils']` がDXF-viewerの`utils`パッケージで汚染される
    （2026-07-29、フルスイート実行時のみ再現する形で発覚——単体実行では再現しない
    ため見落としやすい）。どちらの場合も、同名の最上位パッケージが既にキャッシュ
    されている可能性があるため、呼び出し前後で確実に退避・復元する。
    """
    saved_modules = {
        name: mod for name, mod in sys.modules.items()
        if name == top_level_package or name.startswith(f'{top_level_package}.')
    }
    for name in saved_modules:
        del sys.modules[name]

    sys.path.insert(0, project_root)
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(project_root)
        for name in list(sys.modules):
            if name == top_level_package or name.startswith(f'{top_level_package}.'):
                del sys.modules[name]
        sys.modules.update(saved_modules)


def _call_diff_manager_compute_label_differences(dxf_path, original_filename):
    """DXF-diff-manager の model.label_diff.compute_label_differences() を呼び出す。"""
    foreign = _import_foreign_module(DIFF_MANAGER_ROOT, 'model', 'model.label_diff')
    _, _, extra_info = foreign.compute_label_differences(
        dxf_path, dxf_path, new_file_original_name=original_filename,
    )
    return extra_info.get('title'), extra_info.get('subtitle')


def _call_visual_diff_compute_label_differences(dxf_path):
    """DXF-visual-diff の utils.label_diff.compute_label_differences() を呼び出す。"""
    foreign = _import_foreign_module(VISUAL_DIFF_ROOT, 'utils', 'utils.label_diff')
    _, _, extra_info = foreign.compute_label_differences(dxf_path, dxf_path)
    return extra_info.get('title'), extra_info.get('subtitle')


def _sample_problem_files():
    if not os.path.isdir(SAMPLE_PROBLEMS_DIR):
        return []
    return sorted(glob.glob(os.path.join(SAMPLE_PROBLEMS_DIR, '*.dxf')))


def _reference_title_subtitle(dxf_path):
    """primary自身の正しい呼び出し方（両オプションをセット）で得られる期待値。"""
    _, info = extract_labels(
        dxf_path,
        extract_drawing_numbers_option=True,
        extract_title_option=True,
        original_filename=os.path.basename(dxf_path),
    )
    return info.get('title'), info.get('subtitle')


@pytest.mark.skipif(
    not _diff_manager_available,
    reason="DXF-diff-manager が見つからないためクロスプロジェクトテストをスキップ",
)
@pytest.mark.parametrize("dxf_path", _sample_problem_files(), ids=lambda p: os.path.basename(p))
def test_title_extraction_matches_diff_manager(dxf_path):
    """DXF-diff-manager の compute_label_differences() が返すTitle/Subtitleが、
    primaryの正しい呼び出し方と一致すること。"""
    expected = _reference_title_subtitle(dxf_path)
    actual = _call_diff_manager_compute_label_differences(dxf_path, os.path.basename(dxf_path))
    assert actual == expected, (
        f"{os.path.basename(dxf_path)}: DXF-diff-managerのTitle/Subtitleがprimaryと不一致。\n"
        f"primary: {expected}\nDXF-diff-manager: {actual}"
    )


# 2026-07-29時点でDXF-visual-diff/utils/label_diff.pyはextract_drawing_numbers_option=True
# を渡していなかったが（18件中7件で誤抽出を確認しxfail(strict=True)としていた）、
# 同日中にprimary側の根本対応（v1.9.22、extract_title_option=Trueであれば内部的に
# タイトルブロックグループ判定を行う設計変更）をextract_labels.pyに同期したところ、
# label_diff.py自体は未変更のままこの不具合が副次的に解消された（xfailがXPASSに
# 転じたため、このparametrize特別扱いを撤去し通常のassertに戻した）。

@pytest.mark.skipif(
    not _visual_diff_available,
    reason="DXF-visual-diff が見つからないためクロスプロジェクトテストをスキップ",
)
@pytest.mark.parametrize("dxf_path", _sample_problem_files(), ids=lambda p: os.path.basename(p))
def test_title_extraction_matches_visual_diff(dxf_path):
    """DXF-visual-diff の compute_label_differences() が返すTitle/Subtitleが、
    primaryの正しい呼び出し方と一致すること。"""
    expected = _reference_title_subtitle(dxf_path)
    actual = _call_visual_diff_compute_label_differences(dxf_path)
    assert actual == expected, (
        f"{os.path.basename(dxf_path)}: DXF-visual-diffのTitle/Subtitleがprimaryと不一致。\n"
        f"primary: {expected}\nDXF-visual-diff: {actual}"
    )
