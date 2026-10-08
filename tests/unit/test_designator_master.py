"""機器符号一覧（designator master）生成・索引ロジックの単体テスト。

`model/designator_master.py` の純粋関数（符号トークンの分解・用途の割り当て・
取り消し線判定・S系除外判定・照合キーの正規化・HMACダイジェスト・索引ロード）を、
合成データのみで検証する。原本 Excel（リポジトリ外・`.xls`）には依存しない
（`dev-workflow` スキル2-5の方針）。

このテストが守る組み合わせは、引き継ぎ書（2026-10-08、設計フェーズ）の
2-1 組み合わせ表に対応する:

- 符号セル行数 × 用途セル行数（`assign_uses`）
- 1行内の分解形: 単一/カンマ/スラッシュ/チルダ標準形/チルダ変則形/空白混入/
  分解形の混在（警告）（`expand_code_cell`）
- 取り消し線: なし/セル全体/リッチテキストの一部範囲（`split_struck_lines`）
- S系除外判定: 用途空/枠取り/実機記述/非S系（`is_spare_entry`）
- 照合キーの正規化: 一覧側の `(CB)` 除去（`master_code_key`）、
  ラベル側のΩ区切り等（`label_match_key`）
- HMACダイジェスト: 鍵による値の安定性・鍵不一致の検出（`code_digest`/
  `key_fingerprint`/`load_index`）
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model.designator_master import (  # noqa: E402
    assign_uses,
    code_digest,
    DesignatorMasterIndex,
    expand_code_cell,
    is_spare_entry,
    key_fingerprint,
    label_match_key,
    load_index,
    master_code_key,
    MasterResult,
    MasterRow,
    spare_exclusion_reason,
    split_struck_lines,
    write_digest_file,
)


# ---------------------------------------------------------------------------
# expand_code_cell — 1行内の分解形
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('token, expected_codes', [
    # 単一トークン（分解不要）
    ('CB001', ['CB001']),
    # カンマ区切り（先頭要素の英字サフィックスを後続要素に適用）
    ('F006C,D,E', ['F006C', 'F006D', 'F006E']),
    ('CP019A,B,C,D', ['CP019A', 'CP019B', 'CP019C', 'CP019D']),
    # 括弧を含むbaseでのカンマ区切り
    ('ELB(CB)066A,B', ['ELB(CB)066A', 'ELB(CB)066B']),
    # スラッシュ区切り
    ('CB361A/B', ['CB361A', 'CB361B']),
    # チルダ範囲・標準形（終端が単一英字）
    ('F008A～K', [f'F008{chr(c)}' for c in range(ord('A'), ord('K') + 1)]),
    # チルダ範囲・変則形（終端が数字付き。末尾の英字1文字だけを終端に使う）
    ('F004A～4I', [f'F004{chr(c)}' for c in range(ord('A'), ord('I') + 1)]),
    # 空白混入: 英字と数字の間
    ('S 163', ['S163']),
    # 空白混入: 括弧と数字の間（原文の規則「英文字と数字の間」より広く除去する）
    ('ELB(CB) 251', ['ELB(CB)251']),
    # 全角→半角（NFKC）。括弧・チルダが全角で来た場合も正規化後に展開できる
    ('Ｆ００８Ａ～K', [f'F008{chr(c)}' for c in range(ord('A'), ord('K') + 1)]),
])
def test_expand_code_cell_known_shapes(token, expected_codes):
    """実データで確認済みの分解形が、期待どおりの符号リストに展開される。"""
    codes, warning = expand_code_cell(token)
    assert codes == expected_codes
    assert warning is None


def test_expand_code_cell_mixed_delimiters_warns():
    """`~` と `,`/`/` が同時に含まれる形（実データ0件）は、分解せず1件のまま
    返し、警告を発生させる（将来の原本更新で出現しても出力から消えないため）。"""
    codes, warning = expand_code_cell('F006A,B～K')
    assert codes == ['F006A,B~K']
    assert warning is not None


def test_expand_code_cell_tilde_without_trailing_letter_warns():
    """`~` の前に英字が無い（範囲の起点が特定できない）形は警告を返す。"""
    codes, warning = expand_code_cell('123～K')
    assert codes == ['123~K']
    assert warning is not None


def test_expand_code_cell_comma_without_letter_suffix_warns():
    """先頭要素に英字サフィックスが無い（base が特定できない）カンマ区切りは
    警告を返す。"""
    codes, warning = expand_code_cell('123,456')
    assert codes == ['123,456']
    assert warning is not None


def test_expand_code_cell_reversed_tilde_range_warns():
    """`~` 範囲の始端・終端が逆順（例: K~A）の場合は警告を返す。"""
    codes, warning = expand_code_cell('F008K～A')
    assert codes == ['F008K~A']
    assert warning is not None


def test_expand_code_cell_empty_token_returns_empty_list():
    """空白のみのトークンは空リストを返す（呼び出し元で捨てられる）。"""
    codes, warning = expand_code_cell('   ')
    assert codes == []
    assert warning is None


# ---------------------------------------------------------------------------
# assign_uses — 符号セル行数 × 用途セル行数
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('code_lines, use_lines, expected_uses', [
    # 符号1行
    (['CB001'], [], ['']),
    (['CB001'], ['AC200V MAIN'], ['AC200V MAIN']),
    # 符号1行・用途複数行 → 連結（決定7, 2026-10-08ユーザー承認）
    (['S10'], ['信号線保護用', '（ELR,ADC,...）'], ['信号線保護用 （ELR,ADC,...）']),
    # 符号複数行・用途0行 → 全符号に空の用途
    (['F006A', 'F006B'], [], ['', '']),
    # 符号複数行・用途1行 → 全符号に同じ用途
    (['F010A,B', 'F010AA'], ['T001/PL001 機器、配線保護用'],
     ['T001/PL001 機器、配線保護用', 'T001/PL001 機器、配線保護用']),
    # 符号複数行・用途が同数 → 行ごとに1対1対応
    (['CB014', 'ELB014'], ['TR001 機器保護用', 'TR001 配線保護用'],
     ['TR001 機器保護用', 'TR001 配線保護用']),
])
def test_assign_uses_known_shapes(code_lines, use_lines, expected_uses):
    """実データで確認済みの符号行数×用途行数の組み合わせが正しく割り当てられる。"""
    uses, warning = assign_uses(code_lines, use_lines)
    assert uses == expected_uses
    assert warning is None


def test_assign_uses_mismatched_multiline_falls_back_and_warns():
    """符号複数行・用途が複数行だが符号と同数でない（実データ0件）場合は、
    用途を連結して全符号に割り当て、警告を返す。"""
    uses, warning = assign_uses(['A001', 'A002', 'A003'], ['用途1', '用途2'])
    assert uses == ['用途1 用途2'] * 3
    assert warning is not None


# ---------------------------------------------------------------------------
# split_struck_lines — 取り消し線（セル全体 / リッチテキストの一部範囲）
# ---------------------------------------------------------------------------

def test_split_struck_lines_no_strike():
    """取り消し線が無いセルは全行 False。"""
    result = split_struck_lines('CB001\nCB002', struck_whole=False)
    assert result == [('CB001', False), ('CB002', False)]


def test_split_struck_lines_whole_cell_struck():
    """セル全体が取り消し線のとき、全行が True になる（実データ: CB201の例）。"""
    result = split_struck_lines('CB201', struck_whole=True)
    assert result == [('CB201', True)]


def test_split_struck_lines_partial_strike_first_line_only():
    """リッチテキストの一部範囲だけが取り消し線で、1行目のみを覆う場合、
    1行目だけが True になる（実データ0件・将来の原本更新に備えた検証）。"""
    text = 'CB001\nCB002'  # オフセット: 'CB001'=0-5, '\n'=5, 'CB002'=6-11
    result = split_struck_lines(text, struck_whole=False, struck_ranges=[(0, 5)])
    assert result == [('CB001', True), ('CB002', False)]


def test_split_struck_lines_partial_strike_second_line_only():
    """2行目だけを覆う取り消し線範囲では、2行目だけが True になる。"""
    text = 'CB001\nCB002'
    result = split_struck_lines(text, struck_whole=False, struck_ranges=[(6, 11)])
    assert result == [('CB001', False), ('CB002', True)]


def test_split_struck_lines_three_lines_middle_struck():
    """3行のうち中央の行だけが取り消し線（実データ r109 のような3行セルを想定）。"""
    text = 'ELB(CB)108\nELB(CB)108\nCP108A'
    # 'ELB(CB)108'=0-10, '\n'=10, 'ELB(CB)108'=11-21, '\n'=21, 'CP108A'=22-28
    result = split_struck_lines(text, struck_whole=False, struck_ranges=[(11, 21)])
    assert result == [('ELB(CB)108', False), ('ELB(CB)108', True), ('CP108A', False)]


# ---------------------------------------------------------------------------
# is_spare_entry / spare_exclusion_reason — S系除外判定
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('code, use, expected', [
    # S系・用途空 → 除外
    ('S123', '', True),
    ('S008', '   ', True),
    # S系・枠取り表記 → 除外
    ('S008', 'F1用', True),
    ('S012', 'B2用', True),
    ('S012', 'B2', True),
    ('S012', 'B1', True),
    # S系・実機記述 → 残す（実データ10件のうちの代表例）
    ('S070', 'MATERIAL LEAK SENSOR', False),
    ('S032', 'CRYO PUMP BAKE HEATER保護用', False),
    ('S010', '信号線、および各種機器類 保護用 （ELR,ADC,LMFM,PFC,...）', False),
    # 非S系 → 除外ルールの対象外（用途が空でも残す。実データ: CB BOX C1 の F023）
    ('F023', '', False),
    ('CB001', 'B1用', False),  # 枠取り表記に見えても非S系なら対象外
])
def test_is_spare_entry(code, use, expected):
    """S系除外ルールはS系符号にのみ適用され、用途が空/枠取り表記のときだけ
    True を返す。非S系符号には一切適用しない。"""
    assert is_spare_entry(code, use) is expected


def test_spare_exclusion_reason_distinguishes_empty_and_frame():
    """除外理由は「用途が空」と「枠取り表記」を区別する。"""
    assert spare_exclusion_reason('') == 'S系・用途空'
    assert spare_exclusion_reason('   ') == 'S系・用途空'
    assert spare_exclusion_reason('F1用') == 'S系・枠取り'


# ---------------------------------------------------------------------------
# master_code_key — 一覧側の照合キー正規化（(CB)除去＝解釈B、2026-10-08決定）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('code, expected', [
    ('ELB(CB)001', 'ELB001'),
    ('ELB(CB)251', 'ELB251'),
    # 括弧を含まない符号は空白除去のみで不変
    ('CB001', 'CB001'),
    ('F023', 'F023'),
    # ELB(CB)005 と既存のプレーン ELB005 が同一キーに統合される（重複排除の根拠）
    ('ELB(CB)005', 'ELB005'),
])
def test_master_code_key(code, expected):
    """一覧の `ELB(CB)001` 形は `(CB)` を除去し `ELB001` として登録する
    （2026-10-08ユーザー決定: 解釈B。原文形は実データのDXFラベルに出現しない
    ため、除去しないと一覧の約31%〈186/602件〉が照合に使われず死蔵する）。"""
    assert master_code_key(code) == expected


def test_master_code_key_of_elb_cb_matches_plain_elb():
    """`ELB(CB)005` と `ELB005` は同じ照合キーに正規化される
    （一覧内の重複が `build_digest_lines()` で自然に1件へ統合される根拠）。"""
    assert master_code_key('ELB(CB)005') == master_code_key('ELB005')


# ---------------------------------------------------------------------------
# label_match_key — DXFラベル側の照合キー正規化
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('label, expected', [
    ('CB002 (15A)', 'CB002'),
    ('S 163', 'S163'),
    # Ω区切り: 空白除去を先に行うと直前空白デリミタが消えて壊れる回帰防護
    ('R0 2.2KΩ', 'R0'),
    ('R84 4.7KΩ', 'R84'),
    ('ELB005 (5A/30mA)', 'ELB005'),
    ('ELB002 (5A/30mA)', 'ELB002'),
    # 候補パターンのみ一致する符号（一覧には無い）もキー自体は正常に取れる
    ('R10', 'R10'),
    ('R10(2.2K)', 'R10'),
])
def test_label_match_key(label, expected):
    """DXFラベルは `normalize_label()` → `_judgment_text()` の順で判定用
    文字列を取り出した後に空白を除去する。順序を逆にすると `R0 2.2KΩ` の
    Ω直前スペースデリミタが先に消えて `'R02.2KΩ'` のような壊れたキーになる
    （この順序がモジュール docstring の固定ルール）。"""
    assert label_match_key(label) == expected


# ---------------------------------------------------------------------------
# code_digest / key_fingerprint — HMACダイジェストの安定性・鍵依存性
# ---------------------------------------------------------------------------

def test_code_digest_stable_for_same_key_and_secret():
    """同じ照合キー・同じ鍵なら常に同じダイジェストになる。"""
    assert code_digest('ELB001', 'secret-a') == code_digest('ELB001', 'secret-a')


def test_code_digest_differs_with_different_secret():
    """鍵が異なれば同じ照合キーでもダイジェストが異なる
    （鍵無しの単純ハッシュでは秘匿にならないため、HMACが必須という設計根拠）。"""
    assert code_digest('ELB001', 'secret-a') != code_digest('ELB001', 'secret-b')


def test_code_digest_differs_with_different_key():
    """同じ鍵でも照合キーが異なればダイジェストが異なる。"""
    assert code_digest('ELB001', 'secret-a') != code_digest('CB001', 'secret-a')


def test_key_fingerprint_differs_with_different_secret():
    """鍵が異なればフィンガープリントも異なる（鍵不一致の検出に使う）。"""
    assert key_fingerprint('secret-a') != key_fingerprint('secret-b')


def test_key_fingerprint_stable_for_same_secret():
    """同じ鍵なら常に同じフィンガープリントになる。"""
    assert key_fingerprint('secret-a') == key_fingerprint('secret-a')


# ---------------------------------------------------------------------------
# load_index / DesignatorMasterIndex — 索引のロードと所属判定
# ---------------------------------------------------------------------------

def _make_result(codes_and_uses):
    rows = [MasterRow('TEST', code, use, 1) for code, use in codes_and_uses]
    return MasterResult(rows=rows)


def test_load_index_returns_none_when_secret_is_empty(tmp_path):
    """鍵が空文字列/Noneのときは `None` を返す（2026-10-08ユーザー決定:
    鍵未設定時はCANのみで継続。DEFが黙って全件0になる状態と区別するため、
    呼び出し側はこの `None` を「一覧なし」として扱う）。"""
    result = _make_result([('ELB001', '用途')])
    digest_path = tmp_path / 'digests.txt'
    write_digest_file(result, digest_path, 'secret-a')
    assert load_index(str(digest_path), '') is None
    assert load_index(str(digest_path), None) is None


def test_load_index_returns_none_when_file_missing(tmp_path):
    """ダイジェストファイルが存在しない場合は `None` を返す。"""
    assert load_index(str(tmp_path / 'missing.txt'), 'secret-a') is None


def test_load_index_returns_none_when_secret_mismatches(tmp_path):
    """ダイジェスト生成時と異なる鍵で読み込むと `None` を返す（鍵不一致を
    フィンガープリントで検出する。黙ってDEFが0件になるのではなく、呼び出し側が
    「鍵が異なる」と判定できるようにするための挙動）。"""
    result = _make_result([('ELB001', '用途')])
    digest_path = tmp_path / 'digests.txt'
    write_digest_file(result, digest_path, 'secret-a')
    assert load_index(str(digest_path), 'secret-wrong') is None


def test_load_index_contains_matches_registered_codes(tmp_path):
    """正しい鍵でロードした索引は、登録済みの符号に対する DXF ラベルを
    `contains()` で True と判定する（`(CB)`除去・Ω区切り等の照合キー正規化が
    一覧側・ラベル側で一致していることの統合確認）。"""
    result = _make_result([('ELB(CB)001', '用途1'), ('CB001', '用途2'), ('F023', '')])
    digest_path = tmp_path / 'digests.txt'
    secret = 'secret-a'
    write_digest_file(result, digest_path, secret)

    index = load_index(str(digest_path), secret)
    assert index is not None
    assert len(index) == 3

    # 一覧にある符号（別称除去・空白混入を含む実データ表記）
    assert index.contains('ELB001') is True
    assert index.contains('ELB001 (5A/30mA)') is True  # (CB)除去後のELB001と一致
    assert index.contains('CB001') is True
    assert index.contains('F023') is True
    # 一覧に無い符号（候補パターンには一致するがDEFにはならない）
    assert index.contains('R10') is False
    assert index.contains('CN3') is False


def test_load_index_fingerprint_attribute_matches_key_fingerprint():
    """ロードした索引の `fingerprint` 属性は `key_fingerprint()` の値と一致する。"""
    result = _make_result([('ELB001', '用途')])
    secret = 'secret-a'
    index = DesignatorMasterIndex(
        digests=frozenset(), fingerprint=key_fingerprint(secret), secret=secret)
    assert index.fingerprint == key_fingerprint(secret)
