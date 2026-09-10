"""端子台矩形の「橋渡し無し・内側CIRCLE」検出の回帰テスト（合成DXF）。

不具合の識別子: 2026-09-10ユーザー報告（EE6312-000-01A.dxf / EE6313-000-01C.dxf
の TBCB001W (17.5A) / TBDCPST (20A) が「端子検出不可」になる）。

以前どう壊れていたか:
  `_build_rect_candidates()` は「CIRCLEによる辺の橋渡しが1箇所以上あること」を
  端子台矩形の必須条件にしていた。しかし上記2つの端子台は、辺のLINEが
  途切れずに閉じており、端子CIRCLE(lineweight=50/color=4)が矩形の**内側**に、
  辺から半径の約1.91倍離れて2列に並ぶ描き方だった。橋渡しが1箇所も無いため
  矩形候補の段階で除外され、対応する矩形が見つからない候補ラベルとして
  `unmatched_labels`（Excel上は「端子検出不可」行）に落ちていた。

  なお「注4)」という付随テキストが両ラベル付近にあったためユーザーはこれを
  原因と推測したが、実際には無関係（別のTEXTエンティティであり、ラベル文字列
  には混入しない）。真因は上記のジオメトリだった。

修正後に保証したいこと:
  bboxの内側にあり、最寄りの辺からの距離が自身の半径の
  `_INSIDE_CIRCLE_EDGE_RATIO`(2.5)倍以内の対象CIRCLEを
  `_MIN_INSIDE_EDGE_CIRCLES`(2)個以上含む閉じた矩形は、橋渡しが無くても
  端子台矩形として採用され、対応するTBラベルが `entries` に含まれ
  `unmatched_labels` には入らないこと。

実データ（`Tools/HostPL-comparer/sample_data/DXF_files/EE6312-000-01A.dxf` ほか）
との突き合わせは実装時に手動で確認済み。本テストは `sample-dxf/` を太らせない
方針（ユーザー選択）に従い、合成DXFのみで検証する。
"""
import os
import sys

import ezdxf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model import terminal_detector as td  # noqa: E402

LW, COLOR = td.TB_LINE_WEIGHT, td.TB_LINE_COLOR
CLW, CCOLOR = td.TB_CIRCLE_WEIGHT, td.TB_CIRCLE_COLOR


def _build_doc_with_inside_circle_rectangle():
    """EE6312-000-01A.dxf の TBCB001W 端子台を模した合成DXFを構築する。

    矩形は4辺が途切れずに閉じており（橋渡し無し）、端子CIRCLE 2個を
    左辺寄り（辺からの距離が半径の1.5倍、_INSIDE_CIRCLE_EDGE_RATIOの2.5倍
    以内）に配置する。ラベルは矩形の直上（primary側）に置く。
    """
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_text('TITLE', dxfattribs={'insert': (0, 100)})
    msp.add_text('UNIT内結線図', dxfattribs={'insert': (20, 100)})
    msp.add_text('TEST SUBTITLE', dxfattribs={'insert': (20, 90)})

    # 10x10の閉じた矩形（橋渡し無し）: x=[0,10], y=[0,10]
    xl, xr, y0, y1 = 0, 10, 0, 10
    msp.add_line((xl, y0), (xl, y1), dxfattribs={'lineweight': LW, 'color': COLOR})
    msp.add_line((xr, y0), (xr, y1), dxfattribs={'lineweight': LW, 'color': COLOR})
    msp.add_line((xl, y0), (xr, y0), dxfattribs={'lineweight': LW, 'color': COLOR})
    msp.add_line((xl, y1), (xr, y1), dxfattribs={'lineweight': LW, 'color': COLOR})

    # 端子CIRCLE 2個（左辺寄り、辺から半径1.5倍の距離）
    msp.add_circle((1.5, 3), 1.0, dxfattribs={'lineweight': CLW, 'color': CCOLOR})
    msp.add_circle((1.5, 7), 1.0, dxfattribs={'lineweight': CLW, 'color': CCOLOR})

    # ラベルは矩形の直上（primary、距離4）
    y_label = y1 + 4
    msp.add_text('TBXX001 (17.5A)', dxfattribs={'insert': (xl + 3, y_label)})

    # 矩形内の端子番号
    msp.add_text('1', dxfattribs={'insert': (xl + 3, 5)})
    msp.add_text('2', dxfattribs={'insert': (xl + 3, 5.1)})

    return doc


def test_closed_rectangle_with_inside_circles_is_detected():
    doc = _build_doc_with_inside_circle_rectangle()
    path = '/tmp/_test_terminal_inside_circle.dxf'
    doc.saveas(path)
    try:
        result = td.analyze_dxf_terminals(path, original_filename='sample.dxf')
    finally:
        os.remove(path)

    assert result['is_target']
    assert result['unmatched_labels'] == [], (
        'TBXX001が「端子検出不可」扱いになっている（橋渡し無し・内側CIRCLE'
        'ケースの回帰）')
    assert len(result['entries']) == 1
    assert result['entries'][0]['label'] == 'TBXX001 (17.5A)'
    assert sorted(result['entries'][0]['numbers']) == [1, 2]


def test_closed_rectangle_without_edge_near_circles_stays_unmatched():
    """内側CIRCLEが辺から遠い（中央寄り）場合は、従来どおり除外され
    unmatched のままであること（既存の領域境界枠等との誤検出防止を壊さない
    ことの確認）。"""
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_text('TITLE', dxfattribs={'insert': (0, 100)})
    msp.add_text('UNIT内結線図', dxfattribs={'insert': (20, 100)})

    xl, xr, y0, y1 = 0, 10, 0, 10
    msp.add_line((xl, y0), (xl, y1), dxfattribs={'lineweight': LW, 'color': COLOR})
    msp.add_line((xr, y0), (xr, y1), dxfattribs={'lineweight': LW, 'color': COLOR})
    msp.add_line((xl, y0), (xr, y0), dxfattribs={'lineweight': LW, 'color': COLOR})
    msp.add_line((xl, y1), (xr, y1), dxfattribs={'lineweight': LW, 'color': COLOR})

    # 中央寄り（辺からの距離が半径の5倍）のCIRCLE
    msp.add_circle((5, 4), 1.0, dxfattribs={'lineweight': CLW, 'color': CCOLOR})
    msp.add_circle((5, 6), 1.0, dxfattribs={'lineweight': CLW, 'color': CCOLOR})

    msp.add_text('TBXX002', dxfattribs={'insert': (xl + 3, y1 + 4)})

    path = '/tmp/_test_terminal_centered_circle.dxf'
    doc.saveas(path)
    try:
        result = td.analyze_dxf_terminals(path, original_filename='sample.dxf')
    finally:
        os.remove(path)

    assert result['entries'] == []
    assert result['unmatched_labels'] == ['TBXX002']
