"""config.py（領域検出パラメータ）と app.py の配線を検証する回帰テスト。

**経緯（2026-09-10）**: 「領域検出の詳細設定」UI（st.expander + st.form、
フィールド11個 + 「設定完了」ボタン）を撤去し、`config.py` の
`region_detection_config` を編集する方式に統一した（ユーザー指示。
DXF-diff-manager の config.py 方式に倣う）。本ファイルはこの変更が
以下を壊していないことを守る:

  1. `config.py` が既定値のときに組み立てる `region_cfg` は
     `DEFAULT_REGION_CONFIG` と完全に等価である（＝UIを外しても検出結果が
     変わらないことの担保）。
  2. `config.py` の追加により、共有ファイル `model/extract_labels.py` の
     `from config import extraction_config` フォールバック挙動が変わって
     いない（`config.py` は意図的に `extraction_config` を定義しない）。
  3. `app._build_region_cfg()` がカスタム値を正しく `region_cfg` へ反映する
     （UI呼び出し元3経路: 「領域を検出」ボタン・「ラベルを抽出」ボタン・
     app() 内の region_cfg 変数が同一の組み立て結果を共有する設計のため、
     組み立て関数自体の正しさを検証すれば3経路とも担保される）。
"""
import importlib
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model.region_detector import DEFAULT_REGION_CONFIG  # noqa: E402


def test_default_region_cfg_equals_default_region_config():
    """config.py が既定値のとき、app._build_region_cfg() の返り値は
    DEFAULT_REGION_CONFIG と完全に等価（UI撤去による検出結果の変化がないことの担保）。"""
    from app import _build_region_cfg
    assert _build_region_cfg() == DEFAULT_REGION_CONFIG


def test_config_module_does_not_define_extraction_config():
    """config.py は意図的に extraction_config を定義しない（定義すると
    model/extract_labels.py の図番抽出パターン等が意図せず変わってしまう。
    他プロジェクトとバイト一致の共有ファイルのため extract_labels.py 自体は
    変更できない）。"""
    import config
    assert not hasattr(config, 'extraction_config')


def test_extract_labels_extraction_config_still_uses_fallback():
    """config.py が存在していても、model/extract_labels.py の
    `extraction_config` は内部フォールバック定義のまま（ImportErrorで
    フォールバックする設計）であることを固定する。"""
    # extract_labels は import 時に config.py の有無を判定するため、
    # 既にキャッシュされていても再import して現在の config.py の状態で確認する。
    import model.extract_labels as extract_labels_module
    importlib.reload(extract_labels_module)
    ec = extract_labels_module.extraction_config
    assert ec.DRAWING_NUMBER_PATTERN == r'[A-Z]{2}\d{4}-\d{3}(?:-\d{2})?[A-Z]'
    assert ec.SOURCE_LABEL_PROXIMITY == 80
    assert ec.DWG_NO_LABEL_PROXIMITY == 80
    assert ec.TITLE_PROXIMITY_X == 80


def test_build_region_cfg_reflects_custom_config_values():
    """region_detection_config の値を変更すると _build_region_cfg() の
    返り値に反映されることを固定する（設定変更が実際に効くことの担保）。
    テスト後に既定値へ戻す（他テストへの副作用防止）。"""
    from app import _build_region_cfg
    import config

    rc = config.region_detection_config
    orig = {
        'FRAME_LINEWEIGHT': rc.FRAME_LINEWEIGHT,
        'AREA_RATIO_PERCENT': rc.AREA_RATIO_PERCENT,
        'NAME_EXCLUDE_TERMS': rc.NAME_EXCLUDE_TERMS,
        'NAME_FILTER_PREFIXES': rc.NAME_FILTER_PREFIXES,
    }
    try:
        rc.FRAME_LINEWEIGHT = 150
        rc.AREA_RATIO_PERCENT = 20
        rc.NAME_EXCLUDE_TERMS = ('FOO', 'BAR')
        rc.NAME_FILTER_PREFIXES = ('SYS',)

        cfg = _build_region_cfg()
        assert cfg['frame_lineweight'] == 150
        assert cfg['area_ratio'] == 0.20
        assert cfg['name_exclude_terms'] == ('FOO', 'BAR')
        assert cfg['name_filter_prefixes'] == ('SYS',)
        # 変更していないキーは DEFAULT_REGION_CONFIG のまま
        assert cfg['region_lineweight'] == DEFAULT_REGION_CONFIG['region_lineweight']
    finally:
        for k, v in orig.items():
            setattr(rc, k, v)


def test_advanced_overrides_can_override_internal_defaults():
    """ADVANCED_OVERRIDES は DEFAULT_REGION_CONFIG の任意のキー（11設定に
    含まれないアルゴリズム内部パラメータを含む）を上書きできる。既定は空辞書
    （アルゴリズム内部値は変更しない）ことも併せて確認する。"""
    from app import _build_region_cfg
    import config

    rc = config.region_detection_config
    assert rc.ADVANCED_OVERRIDES == {}, (
        "既定では ADVANCED_OVERRIDES は空でなければならない"
        "（アルゴリズム内部値を変更しないことの担保）")

    orig = dict(rc.ADVANCED_OVERRIDES)
    try:
        rc.ADVANCED_OVERRIDES = {'face_snap': 0.2}
        cfg = _build_region_cfg()
        assert cfg['face_snap'] == 0.2
        # ADVANCED_OVERRIDES で明示的に指定していないキーはDEFAULT_REGION_CONFIGのまま
        assert cfg['merge_level_tol'] == DEFAULT_REGION_CONFIG['merge_level_tol']
    finally:
        rc.ADVANCED_OVERRIDES = orig
