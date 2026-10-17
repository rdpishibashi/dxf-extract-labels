"""ラベルの3分類（機器符号 / 領域名 / どちらでもない）の仕様（2026-10-10、ユーザー決定）。

**受入条件**: `classify_label()` が次の判定順で分類し、Excelの「機器符号候補」列
（`is_ref_designator_label`）と領域名候補フィルタ（`region_detector`）が同じ関数を使う。
  1. 空・単一英大文字・末尾 `+`/`-` → OTHER
  1-00. ひらがな・カタカナ・漢字を含む（括弧内も含む）→ OTHER（2026-10-17。同日の最初の指定「括弧内は無視」を上書き）
  別称除去: `ELB(CB)001` のような先頭の「符号(別称)番号」形は別称を外して `ELB001` として判定（2026-10-17）
  1b. `CN`+ハイフンを含む・英字+数字2桁・`A`+数字1桁 → REGION_NAME（2026-10-16 追加）
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
    ('U1', DESIGNATOR), ('MC-IF2', DESIGNATOR), ('THM001', DESIGNATOR),
    ('RACK1', DESIGNATOR), ('ECATIN2', DESIGNATOR), ('X14A', DESIGNATOR),
    ('U1(2.2K)', DESIGNATOR), ('U1(1234)', DESIGNATOR),
    ('CP004 (10A)', DESIGNATOR), ('CP004(10A)', DESIGNATOR),
    ('SX001 (NX-SIH400)', DESIGNATOR),
    ('AB12-34', DESIGNATOR), ('W No.M06004', DESIGNATOR), ('W No.M06008(CC050VZRB)', DESIGNATOR),
    # 1a. OTHER_PATTERNS: A・C・L・Q・U・F 以外の英大文字1字+数字1〜2桁は機器符号候補にも領域名候補にもしない
    # （2026-10-10 ユーザー指定。戻すときは OTHER_PATTERNS を空にする）
    ('R10', OTHER), ('R1', OTHER), ('N24', OTHER), ('P24', OTHER), ('D4', OTHER),
    ('X05', OTHER), ('Y02', OTHER), ('M10', OTHER), ('B12', OTHER), ('R10(2.2K)', OTHER),
    ('AWG12', OTHER), ('AWG', OTHER), ('AWG20 (x)', OTHER), ('AWGX1', DESIGNATOR),   # AWG+数字のみ除外
    ('C1', DESIGNATOR), ('L5', DESIGNATOR), ('Q1', DESIGNATOR), ('U2', DESIGNATOR), ('F7', DESIGNATOR),
    ('R100', DESIGNATOR), ('R10A', DESIGNATOR),   # 3桁・後ろに英字は対象外
    # 1b. FORCED_REGION_NAME_PATTERNS（2026-10-16 ユーザー指定）: 追加機器符号に一致しても領域名候補
    #   CN で始まりハイフンを含む
    ('CN-IF2-1', REGION_NAME), ('CN-IF2', REGION_NAME), ('CN-IF3-1A', REGION_NAME), ('ＣＮ－ＩＦ２－１', REGION_NAME),
    ('CN-IF2-', OTHER),   # 末尾ハイフンは従来どおり OTHER
    ('CN1', DESIGNATOR), ('CNPG01A', DESIGNATOR),   # ハイフンなしの CN.* は機器符号のまま
    #   英大文字1字以上+数字ちょうど2桁（追加機器符号 CN.*・LS.*・DCPS.*・TH.* 等も含む）
    ('THM01', REGION_NAME), ('SX01 (NX-SIH400)', REGION_NAME), ('SX03 ** (NX-SID800)', REGION_NAME),
    ('LED16', REGION_NAME), ('CNE06', REGION_NAME), ('LS10', REGION_NAME), ('DCPS01', REGION_NAME),
    ('A10', REGION_NAME), ('U10', REGION_NAME), ('U23', REGION_NAME), ('U10(2.2K)', REGION_NAME),
    ('R10', OTHER), ('M10', OTHER),   # OTHER_PATTERNS が先に判定される
    #   A+数字1桁
    ('A1', REGION_NAME), ('A9', REGION_NAME),
    # 1-0. TB で始まるものはすべて機器符号候補（2026-10-17 ユーザー指定。端子台）。末尾 +/- だけは OTHER
    ('TB', DESIGNATOR), ('TB01', DESIGNATOR), ('TB-1', DESIGNATOR), ('TB 1', DESIGNATOR),
    ('ＴＢ０１', DESIGNATOR), ('TB10(2.2K)', DESIGNATOR), ('TB1+', OTHER), ('TB1-', OTHER),
    # ひらがな・カタカナ・漢字を含むものは（括弧内も含め）機器符号候補でも領域名でもない（2026-10-17 ユーザー指定）
    ('TB取付板', OTHER), ('TB01あ', OTHER), ('DC PS3のみピン番号変更', OTHER), ('MOVE 設定', OTHER),
    ('CP004 (日本語)', OTHER),        # 括弧内の日本語も OTHER（2026-10-17。旧: 判定に使わず DESIGNATOR）
    ('TBBit (下図は参考例)', OTHER), ('AWG24(シールド線)', OTHER), ('Han（ハーティング）コネクタ', OTHER),
    ('TB COMPONENT', DESIGNATOR), ('TBHPB TBHCH', DESIGNATOR),   # 英語の複合語は候補のまま
    # 別称除去（2026-10-17）。後ろが数字で始まる形だけが対象
    ('ELB(CB)001', DESIGNATOR), ('ＥＬＢ（ＣＢ）００１', DESIGNATOR), ('ELB(CB)001 (5A/30mA)', DESIGNATOR),
    ('ELB(CB) 251', DESIGNATOR), ('MCCB(CB)001', DESIGNATOR), ('ELB(CB)', REGION_NAME), ('ABC(DEF)GHI', REGION_NAME),
    ('MSS(MOTOR)', REGION_NAME), ('FB(WH)', REGION_NAME), ('GND(M4)', OTHER), ('R10(2.2K)', OTHER),
    ('SENSOR ユニット', OTHER), ('ユニット', OTHER),   # カタカナも日本語（2026-10-17 ユーザー決定。以前は領域名候補）
    # 3. 英字のみ
    ('CTC', REGION_NAME),                                     # unit_names
    ('GND', OTHER), ('AGND', OTHER), ('MOTOR', OTHER), ('OPEN', OTHER),
    ('APPRV', OTHER), ('XPID', OTHER), ('XPUMPON', OTHER),   # 除外語・除外パターン
    ('SYSTEM', OTHER), ('WHITE', OTHER), ('TITLE', OTHER),
    ('CNESCD', DESIGNATOR), ('CNMDSBD', DESIGNATOR), ('MCBHPB', DESIGNATOR),  # 追加機器符号
    ('MFFX', REGION_NAME), ('FB', REGION_NAME), ('DWGNO', REGION_NAME), ('KURIHARA', REGION_NAME),
    # 「英字+数字+英字+数字」は領域名候補（NON_DESIGNATOR_PATTERNS、2026-10-10）。ただし CN.*・DIO.* 等の
    # 追加機器符号に一致するものは機器符号のまま
    ('DC19A4', REGION_NAME), ('ＤＣ２１Ａ４', REGION_NAME), ('P24F5', REGION_NAME),
    ('SX0201T1', REGION_NAME), ('XLS1A51', REGION_NAME),
    ('ＦＬ１Ｆ① ＊＊ （ＦＬ１Ｆ－Ｈ１２ＲＣＥ）', REGION_NAME),  # FL1F1 の形（以前は機器符号）
    ('CN200CBH1', DESIGNATOR), ('DIO0A4', DESIGNATOR), ('CIR0C4', DESIGNATOR),
    # 「符号 (型式)」でない括弧付き: 括弧より前が英字のみの除外語なら OTHER
    ('GND(M4)', OTHER),
    # 4. 複数語・記号入り
    ('SYSTEM I/F BOX', REGION_NAME), ('EFEM UPPER', REGION_NAME), ('M2 DRV', REGION_NAME),
    ('MPD RACK2 PANEL LIMIT', REGION_NAME),
])
def test_classify_label(label, expected):
    assert classify_label(label) == expected


@pytest.mark.parametrize('label', ['U1', 'CNESCD', 'CP004 (10A)'])
def test_excel_candidate_column_follows_designator_class(label):
    assert is_ref_designator_label(label) is True


@pytest.mark.parametrize('label', ['MFFX', 'GND', 'SYSTEM I/F BOX', 'H'])
def test_excel_candidate_column_rejects_other_classes(label):
    assert is_ref_designator_label(label) is False


_KW = dict(min_letters=3, exclude_lowercase=True, exclude_terms=('NOTE', '☆'),
           exclude_circuit_symbols=True, circuit_keep_terms=('RACK',))


@pytest.mark.parametrize('label, valid', [
    ('MFFX', True), ('CTC', True), ('SYSTEM I/F BOX', True), ('EFEM UPPER', True),
    ('GND(M4)', False), ('CP004 (10A)', False), ('SX001 (NX-SIH400)', False),
    ('THM001', False), ('CNESCD', False), ('GND', False), ('SYSTEM', False),
    ('THM01', True), ('SX01 (NX-SIH400)', True), ('CN-IF2-1', True),   # 2026-10-16 領域名候補化
    ('A10', False),   # 領域名候補だが英字3字未満
    ('RACK1', True),  # circuit_keep_terms（RACK）は機器符号扱いにしない
])
def test_region_name_filter_uses_same_classification(label, valid):
    """領域名候補フィルタは REGION_NAME だけを通す（Excelの候補列と同じ分類を共有）。"""
    assert _is_valid_name_candidate(label, **_KW) is valid
