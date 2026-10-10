"""ラベルの3分類（機器符号 / 領域名 / どちらでもない）の仕様（2026-10-10、ユーザー決定）。

**受入条件**: `classify_label()` が次の判定順で分類し、Excelの「機器符号候補」列
（`is_ref_designator_label`）と領域名候補フィルタ（`region_detector`）が同じ関数を使う。
  1. 空・単一英大文字・末尾 `+`/`-` → OTHER
  2. 英字のみ以外の候補パターンに一致（括弧・` **` より前で判定）→ DESIGNATOR
  3. 英字のみ: unit_names → REGION_NAME／他の除外語・除外パターン → OTHER／
     追加機器符号（CN.*・MC.*・MOT.* 等）→ DESIGNATOR／それ以外 → REGION_NAME
  4. 上記以外（複数語・記号入り）→ REGION_NAME
旧 `ref_designator_patterns.py` の旧判定（`classify_judgment_detailed` 等）が守っていた
境界ケース（`W No.`・`R10(1234)`・`DWGNO` 等）はここに3分類で移し替えた。
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from model.label_classifier import (  # noqa: E402
    classify_label, is_ref_designator_label, DESIGNATOR, REGION_NAME, OTHER)
from model.region_detector import _is_valid_name_candidate  # noqa: E402


@pytest.mark.parametrize('label, expected', [
    # 1. OTHER（単一英字・末尾符号・空）
    ('H', OTHER), ('V', OTHER), ('N24-', OTHER), ('L1+', OTHER), ('', OTHER),
    # 2. DESIGNATOR（英字のみ以外の候補パターン。括弧・型式は無視して判定）
    ('R10', DESIGNATOR), ('CN-IF2-1', DESIGNATOR), ('THM01', DESIGNATOR),
    ('RACK1', DESIGNATOR), ('ECATIN2', DESIGNATOR), ('X14A', DESIGNATOR),
    ('R10(2.2K)', DESIGNATOR), ('R10(1234)', DESIGNATOR),
    ('CP004 (10A)', DESIGNATOR), ('CP004(10A)', DESIGNATOR),
    ('SX01 (NX-SIH400)', DESIGNATOR), ('ＦＬ１Ｆ① ＊＊ （ＦＬ１Ｆ－Ｈ１２ＲＣＥ）', DESIGNATOR),
    ('AB12-34', DESIGNATOR), ('W No.M06004', DESIGNATOR), ('W No.M06008(CC050VZRB)', DESIGNATOR),
    # 3. 英字のみ
    ('CTC', REGION_NAME),                                     # unit_names
    ('GND', OTHER), ('AGND', OTHER), ('MOTOR', OTHER), ('OPEN', OTHER),
    ('APPRV', OTHER), ('XPID', OTHER), ('XPUMPON', OTHER),   # 除外語・除外パターン
    ('SYSTEM', OTHER), ('WHITE', OTHER), ('TITLE', OTHER),
    ('CNESCD', DESIGNATOR), ('CNMDSBD', DESIGNATOR), ('MCBHPB', DESIGNATOR),  # 追加機器符号
    ('MFFX', REGION_NAME), ('FB', REGION_NAME), ('DWGNO', REGION_NAME), ('KURIHARA', REGION_NAME),
    # 「符号 (型式)」でない括弧付き: 括弧より前が英字のみの除外語なら OTHER
    ('GND(M4)', OTHER),
    # 4. 複数語・記号入り
    ('SYSTEM I/F BOX', REGION_NAME), ('EFEM UPPER', REGION_NAME), ('M2 DRV', REGION_NAME),
    ('MPD RACK2 PANEL LIMIT', REGION_NAME),
])
def test_classify_label(label, expected):
    assert classify_label(label) == expected


@pytest.mark.parametrize('label', ['R10', 'CNESCD', 'CP004 (10A)'])
def test_excel_candidate_column_follows_designator_class(label):
    assert is_ref_designator_label(label) is True


@pytest.mark.parametrize('label', ['MFFX', 'GND', 'SYSTEM I/F BOX', 'H'])
def test_excel_candidate_column_rejects_other_classes(label):
    assert is_ref_designator_label(label) is False


_KW = dict(min_letters=3, exclude_lowercase=True, exclude_terms=('NOTE', '☆'),
           exclude_circuit_symbols=True, circuit_keep_terms=('RACK',))


@pytest.mark.parametrize('label, valid', [
    ('MFFX', True), ('CTC', True), ('SYSTEM I/F BOX', True), ('EFEM UPPER', True),
    ('GND(M4)', False), ('CP004 (10A)', False), ('SX01 (NX-SIH400)', False),
    ('THM01', False), ('CNESCD', False), ('GND', False), ('SYSTEM', False),
    ('RACK1', True),  # circuit_keep_terms（RACK）は機器符号扱いにしない
])
def test_region_name_filter_uses_same_classification(label, valid):
    """領域名候補フィルタは REGION_NAME だけを通す（Excelの候補列と同じ分類を共有）。"""
    assert _is_valid_name_candidate(label, **_KW) is valid
