"""Shared-pool lookup of sample DXFs: find them **by file name, anywhere under
``Tools/sample-dxf/``** (recursively, including ``dataset/``).

Tests must not hardcode a sample's folder: files and folders move, and a
hardcoded path turns the test into a silent skip (2026-10-26: moving folders
under ``dataset/`` skipped four real-data tests). See ``sample-dxf/README.md``
「テストからのデータの探し方」.

- ``find_sample(name)``: path or ``None``. Search order: pool root, the curated
  folders (fixed order), then the whole tree in sorted order — so a name that
  exists in several folders always resolves to the same copy.
- ``require_sample(name)``: like ``find_sample`` but a missing file is an error
  when the pool exists (a skip would hide missing data). Returns ``None`` only
  when the pool itself is absent (machine without the shared data).
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(PROJECT_ROOT, 'sample-dxf')
CURATED_SUBDIRS = ('problems', 'samples', 'terminal-detector', 'viewer-error')


def pool_exists():
    return os.path.isdir(SAMPLE_DIR)


def find_sample(name):
    direct = os.path.join(SAMPLE_DIR, name)
    if os.path.exists(direct):
        return direct
    for sub in CURATED_SUBDIRS:
        path = os.path.join(SAMPLE_DIR, sub, name)
        if os.path.exists(path):
            return path
    for dirpath, dirnames, filenames in os.walk(SAMPLE_DIR):
        dirnames.sort()
        if name in filenames:
            return os.path.join(dirpath, name)
    return None


def require_sample(name):
    path = find_sample(name)
    if path is None and pool_exists():
        raise AssertionError(
            f'{name} が sample-dxf 配下（再帰）に無い（データ欠落。skip にしない）')
    return path
