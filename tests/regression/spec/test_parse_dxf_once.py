"""仕様: 1ファイルの処理の中で DXF を1回だけ解析する（`doc` 引数）。出力は `doc` の有無で変わらない。

**受入条件**（v3.23.0、2026-11-21。Cable-checker の計測で解析が処理時間の約6割だった）:

- Given 1つの DXF、When `extraction_pipeline.detect_file_regions()` / `extract_file_data()`（領域付き・通常）
  / `terminal_detector.analyze_dxf_terminals()` を呼ぶ、Then `ezdxf.readfile` は**1回だけ**（従来は2回）。
- Given 解析済みの `doc`、When `extract_labels` / `analyze_dxf_regions` / `extract_ref_designator_data` に
  `doc=` を渡す、Then `dxf_file` を読み直さず、戻り値は `doc` を渡さない呼び出しと完全に同じ。
- Given `doc` を渡された関数、When 呼び出した後、Then 文書は書き換えられていない
  （複数の処理が同じ文書を共有するので、1つが書き換えると他の結果が変わる）。
- Given 読めないファイル、When 上の入口を呼ぶ、Then 従来と同じ `error`／警告で終わる（例外にしない）。

境界・例外の意図: `doc=None`（省略）は従来どおり `dxf_file` を読む。パイプラインが読み込みに失敗したときも
`doc=None` を渡し、各関数が自分で読み直して従来の失敗処理をする（エラー文言を変えないため）。
`dxf_file` は `info['filename']` などのために引き続き必要。

実データは `sample-dxf/` を名前で引く（`tests/sample_data.py`）。合成 DXF は `tmp_path`。
"""
import hashlib
import os
import sys

import ezdxf
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tests'))

from model import extraction_pipeline, terminal_detector  # noqa: E402
from model.extract_labels import extract_labels  # noqa: E402
from model.ref_designator import extract_ref_designator_data  # noqa: E402
from model.region_detector import analyze_dxf_regions  # noqa: E402
from sample_data import require_sample  # noqa: E402

REGION_CFG = {'detect_minor_regions': True, 'local_rotation': True}
LABEL_OPTS = dict(extract_drawing_numbers_option=True, extract_title_option=True)


@pytest.fixture
def readfile_calls(monkeypatch):
    """`ezdxf.readfile` の呼び出しパスを記録する（本物に委譲する。全モジュールが `ezdxf.readfile` を使う）。"""
    calls = []
    real = ezdxf.readfile

    def counting(path, *args, **kwargs):
        calls.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(ezdxf, 'readfile', counting)
    return calls


def _synthetic(tmp_path, with_frame=True):
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_text('R10', dxfattribs={'insert': (50, 50), 'height': 3})
    msp.add_text('SYSTEM I/F BOX', dxfattribs={'insert': (60, 20), 'height': 3})
    if with_frame:
        for p1, p2 in [((0, 0), (200, 0)), ((200, 0), (200, 100)),
                       ((200, 100), (0, 100)), ((0, 100), (0, 0))]:
            msp.add_line(p1, p2, dxfattribs={'lineweight': 100, 'color': 7})
    path = str(tmp_path / ('framed.dxf' if with_frame else 'frameless.dxf'))
    doc.saveas(path)
    return path


def _fingerprint(doc):
    """文書の内容の指紋（読み取りだけなら変わらない）。"""
    h = hashlib.md5()
    for layout in doc.layouts:
        for e in layout:
            h.update(f'{layout.name}|{e.dxftype()}|{e.dxf.handle}|{sorted(e.dxfattribs().items(), key=str)}'.encode())
    h.update(f'{len(doc.entitydb)}|{len(doc.blocks)}|{len(doc.layers)}'.encode())
    return h.hexdigest()


# ---- 読み込み回数 -------------------------------------------------------------

@pytest.mark.parametrize('with_frame', [True, False], ids=['framed', 'frameless'])
def test_pipeline_parses_a_synthetic_dxf_once_per_call(tmp_path, readfile_calls, with_frame):
    path = _synthetic(tmp_path, with_frame)
    analysis = extraction_pipeline.detect_file_regions(path, 'a.dxf', REGION_CFG)
    assert len(readfile_calls) == 1
    readfile_calls.clear()
    extraction_pipeline.extract_file_data(path, 'a.dxf', 100, analysis=analysis)   # 領域付きモード
    assert len(readfile_calls) == 1
    readfile_calls.clear()
    extraction_pipeline.extract_file_data(path, 'a.dxf', 100)                      # 通常モード
    assert len(readfile_calls) == 1


@pytest.mark.parametrize('name', ['EE6892-039-02A.dxf', 'EE6892-039-82A.dxf'], ids=['target', 'excluded'])
def test_terminal_detector_parses_real_file_once(readfile_calls, name):
    """対象ファイル（UNIT内結線図）は `extract_labels` と端子台解析で、対象外はタイトル判定だけで読む。どちらも1回。"""
    path = require_sample(name)
    if path is None:
        pytest.skip('sample-dxf が無い環境')
    result = terminal_detector.analyze_dxf_terminals(path, name)
    assert result['error'] is None
    assert len(readfile_calls) == 1


def test_a_passed_doc_is_never_reparsed(tmp_path, readfile_calls):
    """`doc` を渡した3つの入口は `ezdxf.readfile` を一度も呼ばない。"""
    path = _synthetic(tmp_path)
    doc = ezdxf.readfile(path)
    readfile_calls.clear()
    extract_labels(path, doc=doc, **LABEL_OPTS)
    analyze_dxf_regions(path, REGION_CFG, doc=doc)
    extract_ref_designator_data(path, frame_lineweight=100, doc=doc)
    assert readfile_calls == []


# ---- doc の有無で出力が変わらない ------------------------------------------------

def _outputs(path, doc):
    kw = {} if doc is None else {'doc': doc}
    return {
        'labels': extract_labels(path, include_coordinates=True, **LABEL_OPTS, **kw),
        'regions': analyze_dxf_regions(path, REGION_CFG, **kw),
        'ref': extract_ref_designator_data(path, frame_lineweight=100, **kw),
    }


def test_synthetic_outputs_are_identical_with_and_without_doc(tmp_path):
    for with_frame in (True, False):
        path = _synthetic(tmp_path, with_frame)
        assert _outputs(path, None) == _outputs(path, ezdxf.readfile(path))


# 通常の図面（領域あり）／Model Space が空で Paper Space に図形がある図面。枠なしの合成は上の frameless が担う。
REAL_SAMPLES = ['EE6888-604-01A.dxf', 'EE5322-455-01B.dxf']


@pytest.mark.slow
@pytest.mark.parametrize('name', REAL_SAMPLES)
def test_real_outputs_are_identical_with_and_without_doc(name):
    path = require_sample(name)
    if path is None:
        pytest.skip('sample-dxf が無い環境')
    assert _outputs(path, None) == _outputs(path, ezdxf.readfile(path))


# ---- 読み取り専用 ---------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize('name', REAL_SAMPLES)
def test_shared_doc_is_not_modified_and_results_are_stable(name):
    """同じ文書を3つの処理で2巡しても、文書は変わらず、2巡目の結果も1巡目と同じ。"""
    path = require_sample(name)
    if path is None:
        pytest.skip('sample-dxf が無い環境')
    doc = ezdxf.readfile(path)
    before = _fingerprint(doc)
    first = _outputs(path, doc)
    assert _fingerprint(doc) == before
    second = _outputs(path, doc)
    assert _fingerprint(doc) == before
    assert first == second


# ---- 読み込み失敗は従来どおり ------------------------------------------------------

def test_unreadable_file_keeps_the_previous_failure_behavior(tmp_path):
    """壊れたファイルは例外にならず、領域検出は error、抽出は警告つきの空結果、端子一覧は対象外。
    `error` は `ezdxf` の例外文言そのまま（baseline-20261121-docreuse で同じ文言を確認済み）。"""
    path = str(tmp_path / 'broken.dxf')
    with open(path, 'w') as f:
        f.write('not a dxf file\n')
    analysis = extraction_pipeline.detect_file_regions(path, 'broken.dxf', REGION_CFG)
    assert analysis['error'] == f"File '{path}' is not a DXF file."
    assert analysis['regions'] == [] and analysis['title'] is None
    for data in (extraction_pipeline.extract_file_data(path, 'broken.dxf', 100, analysis=analysis),
                 extraction_pipeline.extract_file_data(path, 'broken.dxf', 100)):
        assert data['labels'] == [] and data['warning'] is not None
    terminals = terminal_detector.analyze_dxf_terminals(path, 'broken.dxf')
    assert terminals['is_target'] is False and terminals['error'] is None
