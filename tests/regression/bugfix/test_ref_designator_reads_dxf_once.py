"""不具合再発防止: `extract_ref_designator_data()` が同じDXFを2回読み込む。

**識別子**: 2026-10-09、テスト軽量化の計測で発見（`ezdxf.readfile` 呼び出し回数の実測）。

**以前どう壊れていたか（再現条件）**: `collect_in_frame_labels()` が `ezdxf.readfile()`
で読み込んだ後、図面枠が検出できず `collected['error']` が立つと、
`_collect_all_labels_fallback()` が**同じファイルをもう一度** `ezdxf.readfile()` していた。
`EE6892-455B.dxf`（40MB）では読み込みだけで約2.8秒かかり、図面枠なしの図面を
抽出するたびにアプリでも無駄な待ち時間が発生していた。

**修正後に保証したいこと**: 図面枠の有無によらず、`extract_ref_designator_data()` 1回の
呼び出しで `ezdxf.readfile()` は**1回だけ**。結果（ラベル・警告）は従来と同一で、
読み込めないファイルでも例外にならず従来どおり警告つきの空結果を返す。

合成DXF（tmp_path）のみを使う。
"""
import os
import sys

import ezdxf
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from model import ref_designator  # noqa: E402


@pytest.fixture
def readfile_calls(monkeypatch):
    """`ezdxf.readfile` の呼び出しパスを記録する（本物に委譲する）。"""
    calls = []
    real = ezdxf.readfile

    def counting(path, *args, **kwargs):
        calls.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(ref_designator.ezdxf, 'readfile', counting)
    return calls


def _save(tmp_path, with_frame):
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_text('R10', dxfattribs={'insert': (50, 50), 'height': 3})
    if with_frame:
        for p1, p2 in [((0, 0), (200, 0)), ((200, 0), (200, 100)),
                       ((200, 100), (0, 100)), ((0, 100), (0, 0))]:
            msp.add_line(p1, p2, dxfattribs={'lineweight': 100, 'color': 7})
    path = str(tmp_path / ('framed.dxf' if with_frame else 'frameless.dxf'))
    doc.saveas(path)
    return path


def test_frameless_dxf_fallback_reads_file_only_once(tmp_path, readfile_calls):
    """図面枠なし→全ラベル収集のフォールバック経路でも読み込みは1回。結果は従来どおり。"""
    path = _save(tmp_path, with_frame=False)
    data = ref_designator.extract_ref_designator_data(
        path, frame_lineweight=100, original_filename='frameless.dxf')
    assert len(readfile_calls) == 1
    assert data['frames'] == 0
    assert data['warning'] is not None          # フォールバックした旨の警告
    assert 'R10' in {t for t, _x, _y in data['labels']}


def test_framed_dxf_reads_file_only_once(tmp_path, readfile_calls):
    """図面枠ありの通常経路でも読み込みは1回（フォールバックの有無に関わらず）。"""
    path = _save(tmp_path, with_frame=True)
    ref_designator.extract_ref_designator_data(
        path, frame_lineweight=100, original_filename='framed.dxf')
    assert len(readfile_calls) == 1


def test_unreadable_file_still_returns_empty_result_with_warning(tmp_path):
    """読み込めないファイルは例外にせず、警告つきの空結果を返す（従来の挙動の固定）。"""
    data = ref_designator.extract_ref_designator_data(
        str(tmp_path / 'missing.dxf'), frame_lineweight=100,
        original_filename='missing.dxf')
    assert data['labels'] == []
    assert data['total_in_frame'] == 0
    assert data['warning'] is not None
