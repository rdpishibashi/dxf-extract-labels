"""ひらがな・カタカナ・漢字を含むラベルは領域名の候補にしない（2026-10-10 ひらがな・漢字、2026-10-17 カタカナも、ユーザー指定）。

**受入条件**: `_is_valid_name_candidate()` は、NFKC正規化後にひらがな（U+3040-309F）または
漢字（U+3400-4DBF, U+4E00-9FFF）・カタカナ（U+30A0-30FF）を含むラベルを候補にしない（注記文であって領域名ではない。
実例 `EE6333-610-07A.dxf` の `DC PS3のみピン番号変更`）。英数字だけは従来どおり。
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from model.region_detector import _is_valid_name_candidate  # noqa: E402

_KW = dict(min_letters=3, exclude_lowercase=True, exclude_terms=('NOTE', '☆'),
           exclude_circuit_symbols=True, circuit_keep_terms=('RACK',))


@pytest.mark.parametrize('label, valid', [
    ('DC PS3のみピン番号変更', False),       # ひらがな・漢字
    ('AWG24(シールド線)', False),            # 漢字（カタカナ混じり）
    ('SYSTEM PANEL CLOSE signal use (信号使用)', False),   # 括弧内の日本語も候補にしない（2026-10-17）
    ('MOVE 設定', False),
    ('SYSTEM I/F BOX', True),
    ('MFFX', True),
    ('CTRL BOX ＢＯＸ', True),               # 全角英字は NFKC で半角になる（日本語ではない）
    ('SENSOR ユニット', False),              # カタカナも日本語（2026-10-17 ユーザー決定。以前は対象外）
])
def test_hiragana_kanji_excluded_from_region_name_candidates(label, valid):
    assert _is_valid_name_candidate(label, **_KW) is valid
