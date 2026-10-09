"""pytest 共通設定: 重いテストへの `slow` マーカー付与。

背景（2026-10-09、テスト軽量化）:
    フルスイートは約100秒で、その約6割が実DXF（`EE6892-455B.dxf` 40MB 等）の
    解析を伴う十数件のテストに集中している（計測値は同日の `--durations` による）。
    作業中は `pytest tests -m "not slow"` でこれらを除外し、フルスイートは
    コミット前などのゲートでのみ回す運用にする。
    **既定の `pytest tests` は従来どおり全件実行する**（マーカーは選択のための
    目印であり、テストの内容・件数は変えない）。

非パラメータ化テストは各テストに `@pytest.mark.slow` を直接付けている。
パラメータ化テストのうち重い param だけをここで nodeid 指定で付与する
（同じ関数でも軽い param があるため、ファイル名での一括判定はしない）。
"""
import pytest

# 計測で1秒以上かかった param（2026-10-09 baseline `--durations`）。
SLOW_PARAM_NODEIDS = frozenset({
    'tests/regression/test_title_extraction_cross_project.py::test_title_extraction_matches_diff_manager[EE6892-455B.dxf]',
    'tests/regression/test_title_extraction_cross_project.py::test_title_extraction_matches_diff_manager[EE6892-612-01B.dxf]',
    'tests/regression/test_region_naming_cross_project.py::test_region_naming_matches_across_projects[EE6892-455B.dxf]',
    'tests/regression/test_region_naming_cross_project.py::test_region_naming_matches_across_projects[DE3527-553-10E.dxf]',
    'tests/regression/test_region_naming_cross_project.py::test_region_naming_matches_across_projects[DE5434-553-10B.dxf]',
    'tests/regression/test_ref_designator.py::test_is_ref_designator_label_flags_are_consistent_with_classify_labels[EE6868-500-01C.dxf]',
    'tests/regression/test_ref_designator.py::test_titleblock_content_excluded_from_real_sample[EE6868-500-01C.dxf]',
})


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.nodeid in SLOW_PARAM_NODEIDS:
            item.add_marker(pytest.mark.slow)
