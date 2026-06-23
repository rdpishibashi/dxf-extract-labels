"""
図番／流用元図番の判別（determine_drawing_number_types）回帰テスト。

背景:
    旧・現行の2つのタイトルブロック（INSERT）が同一座標に重なっている DXF
    （例: EE6888-602-01A.dxf）で、流用元図番が旧ブロックの値（EE2505-602-26B）
    として誤抽出されていた。正しくは現行ブロックの EE6492-602-02A。
    候補図番に所属ブロックのグループキーを付与し、図番（ファイル名一致）と同じ
    グループ内で流用元図番を判定するよう修正した。

実行:
    cd DXF-extract-labels
    python -m tests.regression.test_drawing_number_types
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.extract_labels import determine_drawing_number_types

# 流用元図番ラベル・DWG No ラベルの実座標（EE6888-602-01A.dxf 由来）
LABELS = [
    ('流用元図番', (647.0, 69.9)),
    ('流用元図番', (646.8, 70.4)),
    ('DWG No.', (724.7, 25.8)),
    ('DWG.No', (665.4, 83.2)),
]


def test_overlapping_blocks_source_from_main_group():
    """重なった旧・現行ブロックで、図番と同じ現行ブロックの流用元を選ぶ。"""
    candidates = [
        ('EE3273-602-01B', (770.0, 25.5), 'OLD'),   # 旧ブロックの図番
        ('EE2505-602-26B', (664.0, 68.2), 'OLD'),   # 旧ブロックの流用元（誤抽出されていた値）
        ('EE6888-602-01A', (770.0, 25.5), 'NEW'),   # 現行ブロックの図番（ファイル名一致）
        ('EE6492-602-02A', (664.0, 68.2), 'NEW'),   # 現行ブロックの流用元（正解）
    ]
    result = determine_drawing_number_types(candidates, all_labels=LABELS, filename='EE6888-602-01A.dxf')
    assert result['main_drawing'] == 'EE6888-602-01A', result
    assert result['source_drawing'] == 'EE6492-602-02A', result


def test_single_block_normal_case():
    """通常（ブロック1つ）の場合は従来通り図番・流用元を判別する。"""
    candidates = [
        ('EE6888-602-01A', (770.0, 25.5), 'G'),
        ('EE6492-602-02A', (664.0, 68.2), 'G'),
    ]
    result = determine_drawing_number_types(candidates, all_labels=LABELS, filename='EE6888-602-01A.dxf')
    assert result['main_drawing'] == 'EE6888-602-01A', result
    assert result['source_drawing'] == 'EE6492-602-02A', result


def test_backward_compatible_two_tuples():
    """グループキーなしの (図番, 座標) 2要素タプルでも動作する（後方互換）。"""
    candidates = [
        ('EE6888-602-01A', (770.0, 25.5)),
        ('EE6492-602-02A', (664.0, 68.2)),
    ]
    result = determine_drawing_number_types(candidates, all_labels=LABELS, filename='EE6888-602-01A.dxf')
    assert result['main_drawing'] == 'EE6888-602-01A', result
    assert result['source_drawing'] == 'EE6492-602-02A', result


def test_single_candidate():
    result = determine_drawing_number_types([('EE6888-602-01A', (770.0, 25.5), 'G')])
    assert result == {'main_drawing': 'EE6888-602-01A', 'source_drawing': None}


def test_empty():
    assert determine_drawing_number_types([]) == {'main_drawing': None, 'source_drawing': None}


def test_source_equals_main_becomes_none():
    """流用元が図番と同一になった場合は None にする。"""
    candidates = [
        ('EE6888-602-01A', (770.0, 25.5), 'G'),
        ('EE6888-602-01A', (664.0, 68.2), 'G'),
    ]
    result = determine_drawing_number_types(candidates, all_labels=LABELS, filename='EE6888-602-01A.dxf')
    assert result['source_drawing'] is None, result


def test_integration_real_dxf_if_present():
    """実ファイルがローカルにあれば extract_labels 経由で検証（無ければスキップ）。"""
    dxf = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'sample-dxf', 'EE6888-602-01A.dxf',
    )
    if not os.path.exists(dxf):
        print("  (skip: EE6888-602-01A.dxf がローカルに無い)")
        return
    from utils.extract_labels import extract_labels
    _, info = extract_labels(dxf, extract_drawing_numbers_option=True,
                             extract_title_option=True, original_filename='EE6888-602-01A.dxf')
    assert info['main_drawing_number'] == 'EE6888-602-01A', info
    assert info['source_drawing_number'] == 'EE6492-602-02A', info


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failures = []
    for t in tests:
        try:
            t(); print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failures.append(t.__name__); print(f"FAIL: {t.__name__}\n      {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(_run_all())
