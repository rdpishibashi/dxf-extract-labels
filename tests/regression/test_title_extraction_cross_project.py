"""DXF-diff-manager との Title/Subtitle抽出クロスプロジェクト一貫性テスト。

**2026-09-18訂正**: 当初はDXF-diff-manager/DXF-visual-diffの両方を対象にしていたが、
DXF-visual-diffが2026-09-18にラベル比較機能自体（`utils/label_diff.py`・
`diff_labels.xlsx`出力）を削除したため、比較対象から除外した（同プロジェクトには
もう`compute_label_differences()`の呼び出しパスが存在しない）。DXF-diff-manager側の
チェックは引き続き有効。

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

DXF-diff-manager が見つからない環境ではスキップする。
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
SAMPLE_PROBLEMS_DIR = os.path.join(TOOLS_ROOT, 'sample-dxf', 'problems')

_diff_manager_available = os.path.isfile(os.path.join(DIFF_MANAGER_ROOT, 'model', 'label_diff.py'))

def _import_foreign_module(project_root, top_level_package, module_name):
    """他プロジェクトのモジュールを、パッケージ名衝突から隔離してインポートする。

    `DXF-diff-manager` は最上位パッケージ名が primary（本プロジェクト）自身と同じ
    `model` で衝突する可能性がある（同名の最上位パッケージが既にキャッシュされている
    ことがあるため）。呼び出し前後で確実に退避・復元する。
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
