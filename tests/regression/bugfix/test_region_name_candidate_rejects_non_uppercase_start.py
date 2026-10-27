"""不具合再発防止: 長い日本語注記文が矩形領域名候補に混入する。

**識別子**: 2026-10-08、ユーザー報告。

**以前どう壊れていたか（再現条件）**: `_is_valid_name_candidate()`
（`model/region_detector.py`）には最小文字数・小文字除外・除外語
（`NOTE`・`☆`）・機器符号判定はあったが、**最大長さ・語頭文字種の制限が
一切なかった**。実データ `EE3273-039-07B.dxf`・`EE3273-039-07C.dxf`
（`/Users/ryozo/Downloads/YO26-0815-2/`、プロジェクト外のユーザー提供DXF）で、
以下の長い注記文が領域名候補として検出されていた:

    「注1) MD SHUTTER I/F B.Dの接地方法（ノイズ対策)について ・<基板上の
    ランド>→<導電性サポ－タ>→<取付板> を介して,接地の事。(配線を用いる
    接地は不可)」

これは日本語の「注」（Note の意）で始まるため、既存の除外語（英単語
`NOTE`）には一致せず素通りしていた。

**修正後に保証したいこと**: 矩形領域名候補は**英大文字で始まる語であること**
を要求する（2026-10-08ユーザー指示。長さによる制限は設けない——
`MD SHUTTER I/F B.D-Ⅵ` のような長い正当な名称まで誤って除外しないため）。

合成データのみで直接 `_is_valid_name_candidate()` を呼び、`DEFAULT_REGION_CONFIG`
の既定値（`name_min_letters`=3・`name_exclude_lowercase`=True・
`name_exclude_terms`=('NOTE','☆')・`exclude_circuit_symbols`=True・
`circuit_symbol_keep_terms`=('RACK',)）と同じ引数で検証する
（実DXFファイルは不要——このバグは文字列入力のみで再現できるため）。

DXF-cross-checker にも同日中にバイト一致で伝播済み
（`Tools/CLAUDE.md`「region_detector.py」節参照）。
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from model.region_detector import _is_valid_name_candidate, DEFAULT_REGION_CONFIG  # noqa: E402

_CFG = DEFAULT_REGION_CONFIG


def _is_valid(t):
    return _is_valid_name_candidate(
        t,
        _CFG['name_min_letters'],
        _CFG['name_exclude_lowercase'],
        _CFG['name_exclude_terms'],
        _CFG['exclude_circuit_symbols'],
        _CFG['circuit_symbol_keep_terms'],
    )


# 実際に報告された注記文（EE3273-039-07B.dxf/07C.dxf より）
_REPORTED_NOTE = (
    '注1) MD SHUTTER I/F B.Dの接地方法（ノイズ対策)について '
    '・<基板上のランド>→<導電性サポ－タ>→<取付板> を介して,接地の事。'
    '(配線を用いる接地は不可)'
)


def test_reported_long_japanese_note_is_rejected():
    """実際に報告された注記文（日本語の「注」で始まる）は候補として無効。"""
    assert _is_valid(_REPORTED_NOTE) is False


def test_correct_region_name_in_same_file_remains_valid():
    """同じ図面に存在する正しい候補（英大文字始まり）は影響を受けない
    （長さ制限を設けていないことの確認——この名称も十分に長い）。"""
    assert _is_valid('MD SHUTTER I/F B.D-Ⅵ') is True


@pytest.mark.parametrize('text, expected', [
    # 既存の有効な領域名パターン（回帰がないことの確認）
    ('RACK1', True),
    ('SYSTEM I/F BOX', True),
    ('EFEM UPPER', True),
    ('CTC', True),
    ('MFFX', True),
    # 日本語の「注」で始まる注記は常に無効（これが今回のバグの核心）
    ('注2) 別の注記テキストでも同様に無効', False),
    ('注) さらに短い注記', False),
    # 既存の除外語（NOTEで始まる）は引き続き無効
    ('NOTE テキスト', False),
    # 数字で始まる語も無効（英大文字始まりの要求により新たに無効化される副作用。
    # 実データで数字始まりの正当な領域名は確認されていない）
    ('1階フロア', False),
])
def test_various_candidates_against_uppercase_start_rule(text, expected):
    """英大文字で始まる語のみが候補として有効になる（既存の正当な名称は
    維持し、日本語注記・数字始まりの語は無効にする）。"""
    assert _is_valid(text) is expected
