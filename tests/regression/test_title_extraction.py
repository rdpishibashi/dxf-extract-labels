"""
タイトル／サブタイトル抽出（extract_title_and_subtitle）回帰テスト。

背景:
    旧・現行のタイトルブロックが同一座標に重なっている DXF
    （例: EE6888-637-01A.dxf, EE6892-617-01B.dxf）で、
    1) タイトル行とサブタイトル行の Y 差が y_threshold(10.0) 未満のため
       サブタイトル行のグループもトップグループ候補に入り、
    2) min_x の単純 min() タイブレークが浮動小数点ノイズ
       （373.0 vs 373.0000000000002）で決まってサブタイトル行が
       タイトルに選ばれ、タイトル＝サブタイトルの寄せ集めになっていた。
    対策:
    ① min_x が許容誤差内で同一のグループは Y 座標最上段を採用（タイブレーク修正）
    ② 図番の所属グループ（main_drawing_group）内に TITLE がある場合は
       同一グループのラベルのみを候補にする（図番判別と同じグループキー方式）

    v1.7.12: 電気回路図以外（機構部品図等）で、図面枠外に置かれた位置記号
    （F/L/H 等）がタイトル末尾に混入し、図番横の頁数「1/1」（数字のみの
    ラベルが2つ）が誤ってサブタイトルとして採用される不具合を修正。
    対策:
    ③ タイトルブロック（INSERT）内の図面枠（lineweight=100・color=7 の LINE、
       機器符号抽出と同じ識別キー）を検出し、その外側にあるラベルを候補から
       除外（`_titleblock_frame_bbox`）。枠を検出できない図面は従来どおり
       内容ベースの判定にフォールバック。
    ④ 数字のみのラベルは常に候補から除外（`_is_titleblock_noise_label`）。
    ⑤ タイトルとサブタイトルが同一内容になった場合はサブタイトルなしとする
       安全策を追加。

実行:
    cd DXF-extract-labels
    python -m tests.regression.test_title_extraction
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.extract_labels import extract_title_and_subtitle

# EE6888-637-01A.dxf のタイトルブロック周辺ラベル（実座標・実グループ構成を再現）
# OLD = 旧ブロック(JZB_0051/61E), NEW = 現行ブロック(JZB_0001/62A), FRM = 枠ブロック(61F)
EE6888_LABELS = [
    ('TITLE', (338.82684200000017, 43.87284900000003), 'OLD'),
    ('REVISION', (373.0000000000002, 48.5), 'OLD'),
    ('ELECTRICAL SCHEMATIC DIAGRAM', (373.0000000000002, 36.64), 'OLD'),
    ('SENSOR LINE', (373.0000000000002, 27.28), 'OLD'),
    ('EE3273-637-02B', (359.0, 15.5), 'OLD'),
    ('TITLE', (338.8268420000001, 43.87284900000003), 'NEW'),
    ('REVISION', (373.0000000000001, 48.5), 'NEW'),
    ('ELECTRICAL SCHEMATIC DIAGRAM', (373.0, 36.64), 'NEW'),
    ('ＳＥＮＳＯＲ ＬＩＮＥ', (373.0, 27.28), 'NEW'),
    ('EE6888-637-01A', (359.0, 15.5), 'NEW'),
    ('Ｆ', (413.0, 29.66), 'NEW'),     # 現行ブロックが持つ枠位置記号
    ('８', (388.75, 7.13), 'NEW'),
    ('1', (406.25, 13.22), 'NEW'),
    ('F', (413.0, 24.88), 'FRM'),      # 枠ブロックの位置記号
    ('8', (393.75, 7.13), 'FRM'),
]
EE6888_DNS = [
    ('EE3273-637-02B', (359.0, 15.5), 'OLD'),
    ('EE6888-637-01A', (359.0, 15.5), 'NEW'),
]


def test_overlapping_blocks_title_and_subtitle():
    """重なった旧・現行ブロックで、現行ブロックのタイトル・サブタイトルを抽出する。"""
    result = extract_title_and_subtitle(EE6888_LABELS, EE6888_DNS, main_drawing_group='NEW')
    assert result['title'] == 'ELECTRICAL SCHEMATIC DIAGRAM', result
    assert result['subtitle'] == 'ＳＥＮＳＯＲ ＬＩＮＥ', result


def test_tie_break_prefers_top_row_without_group():
    """グループ情報なしでも、min_x が同値のときは最上段の行をタイトルに選ぶ。

    修正前は浮動小数点ノイズ（373.0 vs 373.0000000000002）の大小で
    サブタイトル行がタイトルに選ばれていた。
    """
    result = extract_title_and_subtitle(EE6888_LABELS, EE6888_DNS)
    assert result['title'] == 'ELECTRICAL SCHEMATIC DIAGRAM', result


def test_group_without_title_falls_back_to_all_labels():
    """図番グループ内に TITLE ラベルが無い場合（直接配置の図面等）は全ラベルで判定。"""
    labels = [
        ('TITLE', (338.8, 43.9), 'BLOCK_A'),
        ('REVISION', (373.0, 48.5), 'BLOCK_A'),
        ('MAIN CONTROL PANEL', (373.0, 36.6), 'BLOCK_A'),
        ('EE1234-567-01A', (359.0, 15.5), 'DIRECT_1'),  # 図番だけ別グループ（直接配置）
    ]
    dns = [('EE1234-567-01A', (359.0, 15.5), 'DIRECT_1')]
    result = extract_title_and_subtitle(labels, dns, main_drawing_group='DIRECT_1')
    assert result['title'] == 'MAIN CONTROL PANEL', result


def test_backward_compatible_two_tuples():
    """(テキスト, 座標) の2要素タプル・グループ指定なしでも従来どおり動作する。"""
    labels = [
        ('TITLE', (338.8, 43.9)),
        ('REVISION', (373.0, 48.5)),
        ('ELECTRICAL SCHEMATIC DIAGRAM', (373.0, 36.6)),
        ('SENSOR LINE', (373.0, 20.0)),  # y_threshold(10.0) より下の通常配置
    ]
    result = extract_title_and_subtitle(labels, None)
    assert result['title'] == 'ELECTRICAL SCHEMATIC DIAGRAM', result
    assert result['subtitle'] == 'SENSOR LINE', result


def test_no_title_label_returns_none():
    result = extract_title_and_subtitle([('FOO', (0.0, 0.0), 'G')], None, main_drawing_group='G')
    assert result == {'title': None, 'subtitle': None}


def test_numeric_only_candidate_excluded_from_subtitle():
    """数字のみのラベル（頁数「1/1」等が別々のTEXTに分かれたもの）はサブタイトル候補にしない。

    背景（v1.7.12）: 図番横の頁数表示（例:「1」「1」の2ラベルが斜線を挟んで
    フラクション状に配置）が、他に候補が無い場合にそのままサブタイトルとして
    採用されてしまっていた（EE2685-335-01D.dxf 等、実データ94件中39件で確認）。
    """
    labels = [
        ('TITLE', (338.8, 43.9)),
        ('REVISION', (373.0, 48.5)),
        ('BLANK PANEL', (373.0, 36.6)),
        ('1', (400.0, 20.0)),
        ('1', (404.0, 16.0)),
    ]
    result = extract_title_and_subtitle(labels, None)
    assert result['title'] == 'BLANK PANEL', result
    assert result['subtitle'] is None, result


def test_title_equal_subtitle_becomes_none():
    """タイトルとサブタイトルが同一内容になった場合はサブタイトルなしとみなす（安全策）。"""
    labels = [
        ('TITLE', (338.8, 43.9)),
        ('REVISION', (373.0, 48.5)),
        ('SYSTEM BOX', (373.0, 37.0)),
        ('SYSTEM BOX', (373.0, 27.0)),  # 同一内容の行が別グループとして混入
    ]
    result = extract_title_and_subtitle(labels, None)
    assert result['title'] == 'SYSTEM BOX', result
    assert result['subtitle'] is None, result


def _find_local(name):
    """ローカル実データ（sample-dxf/・405_展開接続図/・339_Unit内結線図/）を探す。"""
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for root in (os.path.join(base, 'sample-dxf'),
                 os.path.expanduser('~/Dropbox/Workspace/405_展開接続図'),
                 os.path.expanduser('~/Dropbox/Workspace/339_Unit内結線図')):
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if name in filenames:
                return os.path.join(dirpath, name)
    return None


def test_integration_real_dxf_if_present():
    """実ファイルがローカルにあれば extract_labels 経由で検証（無ければスキップ）。"""
    from utils.extract_labels import extract_labels
    expectations = [
        ('EE6888-637-01A.dxf', 'ELECTRICAL SCHEMATIC DIAGRAM', 'ＳＥＮＳＯＲ ＬＩＮＥ'),
        ('EE6892-617-01B.dxf', 'ELECTRICAL SCHEMATIC DIAGRAM', 'ＣＨＩＬＬＥＲ Ｒ１'),
        # v1.7.12: 図面枠外の位置記号（F/L/H）がタイトル末尾に混入し、
        # 頁数「1/1」がサブタイトルとして誤採用されていた不具合の回帰確認
        ('EE2685-335-01D.dxf', 'Ｄ－ＳＵＢ９Ｐ用ブランクパネル', None),
        ('EE2685-475-96A.dxf', 'ア－スバ－（２５×３２０ｍｍ）', None),
        ('EE5322-455-01B.dxf', 'ＳＹＳＴＥＭ＿Ｉ／Ｆ＿ＢＯＸ＿本体（ＢＫ）', None),
    ]
    for name, exp_title, exp_subtitle in expectations:
        dxf = _find_local(name)
        if not dxf:
            print(f"  (skip: {name} がローカルに無い)")
            continue
        _, info = extract_labels(dxf, extract_drawing_numbers_option=True,
                                 extract_title_option=True, original_filename=name)
        assert info['title'] == exp_title, (name, info['title'])
        assert info['subtitle'] == exp_subtitle, (name, info['subtitle'])


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
