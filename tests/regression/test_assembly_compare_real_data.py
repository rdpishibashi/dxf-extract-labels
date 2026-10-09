"""「結線図-組立図を比較」の実データブラックボックステスト。

model層の純関数（select_wiring_files/select_assembly_candidates/compare_assembly）
は tests/unit/test_assembly_compare.py の合成データで検証済み。ここでは
「実DXFファイルを ref_designator.extract_ref_designator_data() /
model.extract_labels.extract_labels() で抽出した結果」が、その先の
assembly_compare パイプラインと矛盾なく繋がることを確認する
end-to-end スモークテスト（`_handle_extract_button` の通常モード分岐と
同じ手順を踏む）。

テストデータ: sample-dxf/組立図・内結線図_ME25-7103-0_ZMR1_339/
（2026-09-13 ユーザー提供。共有カタログ〈problems/samples/terminal-detector/
viewer-error〉には属さない非カタログデータのため、`_find_sample()` の
探索対象外——他プロジェクトで再利用する場合は改めてカタログに登録すること）。
全71件のうち、実行時間を抑えるためA側2件（全角・半角タイトル各1件、
D13の全角/半角混在を実データで固定）＋B側1件（実在する組立図はこの
データセット中1件のみ）の計3件だけを対象にする。

フォルダが無い環境（他マシン・CI）では収集不能ではなくスキップする。
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(
    os.path.dirname(PROJECT_ROOT), 'sample-dxf', '組立図・内結線図_ME25-7103-0_ZMR1_339')
# A側: 全角タイトル1件・半角タイトル1件（実データで確認した表記混在、D13）
A_FILES = ['EE6491-039-16A.dxf', 'EE6491-039-21A.dxf']
# B側: タイトルに「組立図」を含む実在ファイル（このデータセット中1件のみ）
B_FILE = 'EE6892-455B.dxf'

pytestmark = [
    pytest.mark.skipif(
        not all(os.path.exists(os.path.join(DATA_DIR, f)) for f in A_FILES + [B_FILE]),
        reason=f'実データ未配置のためスキップ: {DATA_DIR}'),
    # module fixture `ref_final` が40MBの実DXFを抽出する（setup 約9秒）ため、
    # モジュール全体を slow とする（conftest.py 参照）。
    pytest.mark.slow,
]

from model import ref_designator  # noqa: E402
from model import assembly_compare as ac  # noqa: E402
from tests import dxf_cache  # noqa: E402
from model.excel_output import create_ref_designator_excel_output  # noqa: E402


@pytest.fixture(scope='module')
def ref_final():
    """`_handle_extract_button()` 通常モード分岐と同じ手順で3ファイルを抽出する。

    3ファイルの抽出自体に十数秒かかる（実データ）ため、module スコープで
    1回だけ実行し、このモジュール内の全テストで共有する。"""
    ref_data_by_file = {}
    for fname in A_FILES + [B_FILE]:
        path = os.path.join(DATA_DIR, fname)
        data = dxf_cache.extract_ref_designator_data(
            path, frame_lineweight=100, original_filename=fname)
        _, dn_info = dxf_cache.extract_labels(
            path, extract_drawing_numbers_option=True, extract_title_option=True,
            original_filename=fname)
        data['main_drawing_number'] = dn_info.get('main_drawing_number')
        data['source_drawing_number'] = dn_info.get('source_drawing_number')
        data['title'] = dn_info.get('title')
        data['subtitle'] = dn_info.get('subtitle')
        ref_data_by_file[fname] = data
    return ref_designator.build_ref_designator_final(ref_data_by_file, 'asc')


def test_select_wiring_files_finds_both_fullwidth_and_halfwidth_titles(ref_final):
    """実データのタイトル表記混在（全角『ＵＮＩＴ内結線図』・半角『UNIT内結線図』）
    の両方がA側として選ばれることを固定する（D13）。"""
    assert ref_final['EE6491-039-16A.dxf']['title'] == 'ＵＮＩＴ内結線図'
    assert ref_final['EE6491-039-21A.dxf']['title'] == 'UNIT内結線図'
    assert ac.select_wiring_files(ref_final) == sorted(A_FILES)


def test_select_assembly_candidates_finds_the_single_assembly_drawing(ref_final):
    assert ac.select_assembly_candidates(ref_final) == [B_FILE]


def test_compare_assembly_end_to_end_produces_rows(ref_final):
    """実データでA・B両方にラベルが存在し、比較結果に『両方』の区分が
    含まれること（完全に噛み合わない・空にならないことの確認）。"""
    wiring_files = ac.select_wiring_files(ref_final)
    result = ac.compare_assembly(ref_final, wiring_files, B_FILE)

    assert result['b_file'] == B_FILE
    assert result['summary']['A 対象図番数'] == 2
    assert result['summary']['両方'] > 0
    assert len(result['rows']) == result['summary']['両方'] + \
        result['summary']['A のみ'] + result['summary']['B のみ']


def test_excel_output_includes_assembly_compare_sheet_at_expected_position(ref_final):
    """実データ抽出結果をそのまま `create_ref_designator_excel_output()` に
    渡しても例外なく完走し、比較シートが期待位置（抽出ラベル一覧の直後・
    各ファイルシートの直前）に入ること。"""
    import openpyxl
    from io import BytesIO

    wiring_files = ac.select_wiring_files(ref_final)
    compare_result = ac.compare_assembly(ref_final, wiring_files, B_FILE)

    data = create_ref_designator_excel_output(ref_final, 'asc', assembly_compare=compare_result)
    wb = openpyxl.load_workbook(BytesIO(data))
    assert wb.sheetnames[:3] == ['Summary', '抽出ラベル一覧', '結線図-組立図比較']
    ws = wb['結線図-組立図比較']
    header = [c.value for c in ws[1]]
    assert header == ['機器符号候補', 'ラベル', 'A 合計出現数', 'A 図番数', 'B 出現数', 'A 図番', '比較結果']
    assert ws.max_row - 1 == len(compare_result['rows'])
