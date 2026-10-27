"""ゴールデンテスト: 領域選択の振る舞いを、保存形式に依存しない観測値で固定する。

守るもの（受入条件）:
  Given 合成した解析結果（入れ子・同名候補・別の図面枠・名称なし・除外・エラーのファイル）
  When  seed ごとに乱数で決めた操作列（「抽出領域の選択」の特定・除外、viewer の操作、
        ページ上のチェックボックスの操作）を実際の画面と同じ順序（操作 → 再描画）で30手流す
  Then  各手の後の観測値（出力される領域名・「領域の確認」の対象・5%未満の小領域の採用・
        viewer に渡す表示データ）が、リファクタ前のコードで記録した値と一致する

リファクタ（選択状態の保存形式の変更・モジュール分割）で「振る舞いが変わっていない」ことを
示すための基準であり、記録（`golden/region_selection_golden.json`）は書き換えない。
リファクタで変えてよいのは `_Api` の中身（呼び出し先のモジュール・関数名・引数の渡し方）だけ。
操作列の生成・観測値の取り方・比較は変えない。

記録を作る（リファクタ前のコードでだけ行う）:
  python3 tests/regression/spec/test_region_selection_golden.py --record

記録の更新履歴（振る舞いを意図して変えたときだけ。差が意図した箇所だけであることを確認してから更新する）:
  2026-10-25 v3.17.0: 領域名のハイライトを「その領域の内側の同名ラベル（なければ最も近い1件）」に限定した
    （`region_preview.region_label_occurrences()`）ため、viewer の `geoms` を更新。合成データ d.dxf の
    `cn cs`/`CN CS`（NFKC・大文字で同じ文字列。どちらも領域の外）が、2件から最も近い1件になる。
    確認: `region_label_occurrences` を旧挙動（`resolve_labels`）に差し替えると旧記録と完全に一致し、
    新挙動では gather/targets/unconditional/minor が全60シード・全30手で一致、viewer は最終状態で
    geoms 以外が一致（差は d.dxf の領域1・2・3とその候補だけ）。
"""
import contextlib
import hashlib
import json
import os
import random
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st  # noqa: E402

GOLDEN_PATH = os.path.join(HERE, 'golden', 'region_selection_golden.json')
SEEDS = list(range(60))
STEPS = 30
FRAME_NOT_FOUND = '領域探索を実施することができませんでした: 図面枠が見つかりません'
TYPED_POOL = ['M1 DRV', 'm1 drv', 'ＡＢ１', 'ab1', 'cn cs', 'nope', 'x1', 'Ｘ１', '']


class _Api:
    """呼び出し先をここに集める（リファクタで変えてよいのはこのクラスの中身だけ）。"""

    ANALYSES_KEY = 'region_analyses'

    @staticmethod
    def _mods():
        from model import ref_designator, region_decisions, region_detector, region_preview
        from view import region_confirm_viewer, region_selection
        return (region_confirm_viewer, ref_designator, region_decisions, region_detector,
                region_preview, region_selection)

    @classmethod
    def checked(cls):
        return cls._mods()[5].global_checked_region_names()

    @classmethod
    def excluded(cls):
        return cls._mods()[5].global_excluded_region_names()

    @classmethod
    def pending_cands(cls, reg, checked):
        return cls._mods()[3].pending_candidates_for_region(reg, checked)

    @classmethod
    def candidate_names(cls, analyses):
        return cls._mods()[3].collect_unique_region_candidate_names(analyses)

    @classmethod
    def unnamed_labels(cls, analysis):
        return cls._mods()[1].unnamed_region_labels(analysis)

    @classmethod
    def targets_page(cls, analyses, checked):
        """共有側の対象判定（DXF-cross-checker・HostPL-comparer が使う形）。"""
        return cls._mods()[5].region_confirmation_targets(analyses, checked)

    @classmethod
    def targets_all(cls, analyses, checked):
        """DXF-extract-labels の「領域の確認」の対象（全ファイル）。"""
        return cls._mods()[0].confirmation_targets(analyses, checked)

    @classmethod
    def needs_choice(cls, analysis, pending, checked):
        return cls._mods()[0].needs_choice(analysis, pending, checked)

    @classmethod
    def ensure_choice_defaults(cls, fname, reg, cands, analyses, checked):
        cls._mods()[5].ensure_choice_defaults(fname, reg, cands, analyses, checked)

    @classmethod
    def ensure_typed_for_viewer(cls, fname, analysis):
        from model import region_choices
        rcv = cls._mods()[0]
        for reg, *_r in rcv.decided_regions(analysis) + rcv.unnamed_regions(analysis):
            region_choices.ensure_typed_defaults(st.session_state, fname, reg['id'])

    @classmethod
    def apply_viewer_choice(cls, fname, reg, kind, checked, idx=None, text='', typed_text=None):
        cls._mods()[0].apply_viewer_choice(
            fname, reg, kind, checked, idx=idx, text=text, typed_text=typed_text)

    @classmethod
    def page_cand(cls, fname, reg_id, idx, n_cands, text, checked):
        cls._mods()[5].on_widget_radio(fname, reg_id, idx, n_cands, text, checked)

    @classmethod
    def page_none(cls, fname, reg_id, n_cands, checked):
        cls._mods()[5].on_widget_none(fname, reg_id, n_cands, checked)

    @classmethod
    def gather(cls, fname, analysis):
        return cls._mods()[5].gather_name_selections(fname, analysis)

    @classmethod
    def unconditional(cls, analyses):
        return cls._mods()[5].unconditional_region_names(analyses)

    @classmethod
    def minor(cls, analyses, name_selections, target_names, excluded):
        return cls._mods()[1].add_minor_region_selections(
            analyses, name_selections, target_names, excluded_names=excluded)

    @classmethod
    def adopted(cls, name_selections):
        return cls._mods()[1].adopted_region_names(name_selections)

    @classmethod
    def viewer_data(cls, base, fname, analysis, pending, checked):
        rcv, _rd, _dec, region_detector, region_preview, rs = cls._mods()
        decided = rcv.decided_regions(analysis)
        unnamed = rcv.unnamed_regions(analysis)
        return region_preview.build_confirmation_data(
            base, fname, analysis, pending, decided, st.session_state, checked, False,
            rs.match_typed_region_name, region_detector.pending_candidates_for_region,
            unnamed=unnamed, inactive=rcv.inactive_regions(analysis))

    @classmethod
    def base_render(cls):
        import io
        import ezdxf
        doc = ezdxf.new('R2010')
        doc.modelspace().add_lwpolyline([(0, 0), (500, 0), (500, 500), (0, 500)], close=True)
        buf = io.StringIO()
        doc.write(buf)
        return cls._mods()[4].render_base_svg(buf.getvalue().encode('utf-8'))


# ---------------------------------------------------------------------------
# 合成データ
# ---------------------------------------------------------------------------

def _poly(x, y, w=60, h=60):
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _reg(rid, cands, poly, frame=0, area=12.0, tier=None):
    reg = {'id': rid, 'frame': frame, 'polygon': poly, 'corners': poly, 'area_pct': area,
           'name_candidates': list(cands)}
    if tier is not None:
        reg['default_name_tier'] = tier  # 1/2: 最有力候補が確信度の高い候補（既定は先頭に固定）
    return reg


def _fixtures():
    a = {
        'regions': [
            _reg(0, [(1, 'X1'), (2, 'Y1'), (3, 'Z1')], _poly(0, 0, 100, 100)),
            _reg(1, [(1, 'X1'), (2, 'Y1')], _poly(10, 10, 30, 30)),       # r0 の入れ子（同期しない）
            _reg(2, [(1, 'X1'), (2, 'Y1')], _poly(200, 0)),               # 別位置の同名候補（同期する）
            _reg(3, [(1, 'S1')], _poly(300, 0)),                          # 候補1件
            _reg(4, [(1, 'S1')], _poly(400, 0)),
        ],
        'minor_regions': [_reg(5, [(1, 'S1')], _poly(500, 0, 10, 10), area=2.0)],
        'labels': [('X1', 5, 5), ('Y1', 6, 6), ('Z1', 7, 7), ('S1', 310, 5), ('M1 DRV', 20, 20)],
        'frames': [1], 'main_drawing_number': 'A1',
    }
    b = {
        'regions': [
            _reg(0, [(1, 'X1'), (2, 'Y1'), (3, 'W1')], _poly(0, 0)),
            _reg(1, [(1, 'S1')], _poly(100, 0)),
            _reg(2, [(1, 'P1'), (2, 'Q1')], _poly(200, 0)),
            _reg(3, [(1, 'R1'), (2, 'Q1')], _poly(300, 0), tier=1),   # 後から要選択になる確信度の高い領域
            _reg(4, [(1, 'R1'), (2, 'Q1')], _poly(400, 0)),           # 同じ候補で確信度なし（対比）
            _reg(5, [(1, 'W1'), (2, 'Y1')], _poly(500, 0)),           # 後から要選択になる領域（他で選んだ Y1 を既定にする）
            _reg(6, [(1, 'W1'), (2, 'Y1')], _poly(600, 0), tier=1),   # 同じ候補で確信度あり（既定は先頭に固定）
        ],
        'minor_regions': [],
        'labels': [('X1', 5, 5), ('P1', 205, 5), ('Q1', 305, 5), ('ＡＢ１', 50, 50)],
        'frames': [1], 'main_drawing_number': 'B1',
    }
    c = {
        'regions': [
            _reg(0, [(1, 'M1'), (2, 'N1')], _poly(0, 0), frame=0),
            _reg(1, [(1, 'M1')], _poly(100, 0), frame=1),
            _reg(2, [(1, 'N1'), (2, 'M1')], _poly(200, 0), frame=1),
            _reg(3, [(1, 'N1')], _poly(300, 0), frame=0),
        ],
        'minor_regions': [],
        'labels': [('M1', 5, 5), ('N1', 6, 6)],
        'frames': [1, 2], 'main_drawing_number': 'C1',
    }
    d = {
        'regions': [
            _reg(0, [], _poly(0, 0), area=30.0),
            _reg(1, [], _poly(100, 0), area=8.0),
            _reg(2, [(1, 'CN CS')], _poly(200, 0)),
            _reg(3, [(1, 'CN CS'), (2, 'AA')], _poly(300, 0)),
        ],
        'minor_regions': [],
        'labels': [('M1 DRV', 5, 5), ('ＡＢ１', 6, 6), ('cn cs', 7, 7), ('CN CS', 8, 8), ('AA', 9, 9)],
        'frames': [1], 'main_drawing_number': 'D1',
    }
    return {
        'a.dxf': a, 'b.dxf': b, 'c.dxf': c, 'd.dxf': d,
        'e.dxf': {'regions': [], 'labels': [], 'frames': [], 'error': FRAME_NOT_FOUND},
        'f.dxf': {'regions': [], 'labels': [], 'frames': [], 'error': 'boom'},
        'g.dxf': {'regions': [], 'labels': [], 'frames': [1]},
    }


# ---------------------------------------------------------------------------
# 観測
# ---------------------------------------------------------------------------

def _sel_list(selections):
    return sorted([rid, names] for (_f, rid), names in selections.items())


def _viewable(analyses, checked):
    return [t for t in _Api.targets_all(analyses, checked)
            if not t[1].get('error') and t[1].get('regions')]


def _observe(analyses, base):
    checked = _Api.checked()
    excluded = _Api.excluded()
    gather = {f: _sel_list(_Api.gather(f, analyses[f])) for f in sorted(analyses)}
    by_file = {f: _Api.gather(f, analyses[f]) for f in sorted(analyses)}
    unconditional = sorted(_Api.unconditional(analyses))
    minor_a = _Api.minor(analyses, by_file, unconditional, excluded)
    minor_b = _Api.minor(analyses, by_file, _Api.adopted(by_file), excluded)
    targets = {
        'page': [[f, [r['id'] for r in p]] for f, _a, p in _Api.targets_page(analyses, checked)],
        'all': [[f, [r['id'] for r in p], _Api.needs_choice(a, p, checked)]
                for f, a, p in _Api.targets_all(analyses, checked)],
    }
    viewer = {}
    for f, a, p in _viewable(analyses, checked):
        data = _Api.viewer_data(base, f, a, p, checked)
        viewer[f] = {
            'regions': [
                {**{k: r.get(k) for k in ('key', 'id', 'state', 'name', 'note', 'editable')},
                 'geoms': [g['text'] for g in r.get('geoms', [])]}
                for r in data['regions']],
            'choices': {
                rid: {**ch, 'cands': [{**c, 'geoms': [g['text'] for g in c['geoms']]}
                                      for c in ch['cands']]}
                for rid, ch in data['choices'].items()},
            'locked': data['locked'],
        }
    return {
        'gather': gather, 'targets': targets, 'unconditional': unconditional,
        'minor': {'unconditional': {f: _sel_list(s) for f, s in minor_a.items()},
                  'adopted': {f: _sel_list(s) for f, s in minor_b.items()}},
        'viewer': viewer,
    }


def _digest(obj):
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 操作
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _fresh_state():
    original = st.session_state
    st.session_state = {}
    try:
        yield st.session_state
    finally:
        st.session_state = original


_BASE = {}


def _render(analyses, viewer_file):
    """画面の再描画で行う状態の初期化（`_render_confirmation_file` と同じ順序）。"""
    checked = _Api.checked()
    for fname, analysis, pending in _Api.targets_all(analyses, checked):
        if analysis.get('error') or not analysis.get('regions'):
            continue
        for reg in pending:
            _Api.ensure_choice_defaults(
                fname, reg, _Api.pending_cands(reg, checked), analyses, checked)
        if fname == viewer_file:
            _Api.ensure_typed_for_viewer(fname, analysis)


def _viewer_ops(data):
    """実際の viewer（JS）が送りうる操作。JS が無視する操作は含めない。"""
    ops = []
    regions = {r['id']: r for r in data['regions']}
    for rid, ch in data['choices'].items():
        rid = int(rid)
        r = regions.get(rid)
        if r is None:
            continue
        if r['state'] == 'pending':
            ops += [['cand', rid, c['idx'], c['text']] for c in ch['cands'] if not c['selected']]
            if not ch['none']:
                ops.append(['none', rid])
            if not ch['typed']:
                ops.append(['typed', rid])
            ops += [['typedtext', rid, t] for t in TYPED_POOL]
        elif r['editable']:
            ops += [['typedtext', rid, t] for t in TYPED_POOL if t]
            if ch['typed'] or ch['typedtext']:
                ops.append(['typedoff', rid])
            if not ch['none']:
                ops.append(['editnone', rid])
            else:
                ops.append(['editnoneoff', rid])
            on = not (ch['none'] or (ch['typed'] and not ch['typedMatch']))
            ops.append(['adopt', rid, on, bool(ch['none'])])
    return ops


def _apply_viewer_op(fname, analysis, op, checked):
    kind, rid = op[0], op[1]
    reg = next(r for r in analysis['regions'] if r['id'] == rid)
    if kind == 'cand':
        _Api.apply_viewer_choice(fname, reg, 'cand', checked, idx=op[2], text=op[3])
    elif kind in ('none', 'typed', 'typedoff', 'editnone', 'editnoneoff'):
        _Api.apply_viewer_choice(fname, reg, kind, checked)
    elif kind == 'typedtext':
        _Api.apply_viewer_choice(fname, reg, 'typedtext', checked, typed_text=op[2])
    elif kind == 'adopt':  # 行頭の「採用する」チェック（op[2]=今ON、op[3]=「領域選択しない」がON）
        if op[2]:
            _Api.apply_viewer_choice(fname, reg, 'editnone', checked)
        else:
            _Api.apply_viewer_choice(fname, reg, 'editnoneoff' if op[3] else 'typedoff', checked)


def _step(rng, analyses, names, viewer_file):
    checked = _Api.checked()
    r = rng.random()
    if r < 0.20:
        name = rng.choice(names)
        st.session_state['grc_' + name] = not st.session_state.get('grc_' + name, False)
        return ['grc', name]
    if r < 0.28:
        name = rng.choice(names)
        st.session_state['gre_' + name] = not st.session_state.get('gre_' + name, False)
        return ['gre', name]
    if r < 0.80:
        target = next(((f, a, p) for f, a, p in _viewable(analyses, checked) if f == viewer_file), None)
        if target is None:
            return ['noop']
        fname, analysis, pending = target
        data = _Api.viewer_data(_BASE['base'], fname, analysis, pending, checked)
        ops = _viewer_ops(data)
        if not ops:
            return ['noop']
        kind = rng.choice(sorted({o[0] for o in ops}))  # 種類を先に一様に選ぶ（入力名の候補が多いため）
        op = rng.choice([o for o in ops if o[0] == kind])
        _apply_viewer_op(fname, analysis, op, checked)
        return ['viewer', fname] + op
    pages = [(f, reg, _Api.pending_cands(reg, checked))
             for f, a, p in _viewable(analyses, checked) for reg in p]
    if not pages:
        return ['noop']
    fname, reg, cands = rng.choice(pages)
    on = rng.random() < 0.8
    if rng.random() < 0.7:
        idx = rng.randrange(len(cands))
        st.session_state[f"rcw_{fname}_{reg['id']}_{idx}"] = on
        _Api.page_cand(fname, reg['id'], idx, len(cands), cands[idx][1], checked)
        return ['page_cand', fname, reg['id'], idx, on]
    st.session_state[f"rcw_{fname}_{reg['id']}_none"] = on
    _Api.page_none(fname, reg['id'], len(cands), checked)
    return ['page_none', fname, reg['id'], on]


def run_seed(seed):
    rng = random.Random(seed)
    analyses = _fixtures()
    with _fresh_state() as state:
        _BASE['base'] = _BASE.get('base') or _Api.base_render()
        state[_Api.ANALYSES_KEY] = analyses
        unnamed = sorted({label for a in analyses.values()
                          for label in _Api.unnamed_labels(a).values()})
        names = sorted(_Api.candidate_names(analyses)) + unnamed
        for n in names:
            state['grc_' + n] = True if n in unnamed else rng.random() < 0.75
            state['gre_' + n] = False
        view_files = [None] + sorted(f for f, a in analyses.items()
                                     if a.get('regions') and not a.get('error'))
        viewer_file = rng.choice(view_files)
        _render(analyses, viewer_file)
        steps = []
        obs = None
        for _ in range(STEPS):
            op = _step(rng, analyses, names, viewer_file)
            if rng.random() < 0.15:
                viewer_file = rng.choice(view_files)
            _render(analyses, viewer_file)
            obs = _observe(analyses, _BASE['base'])
            steps.append({'op': op, 'h': {k: _digest(v) for k, v in obs.items()}})
        return {'steps': steps, 'final': obs}


def _load():
    with open(GOLDEN_PATH, encoding='utf-8') as f:
        return json.load(f)


def _canon(obj):
    return json.loads(json.dumps(obj, ensure_ascii=False))


@pytest.mark.parametrize('seed', SEEDS)
def test_selection_behavior_matches_golden_record(seed):
    """守るもの: 同じ操作列を流したとき、各手の観測値が記録と一致する（保存形式に依存しない）。"""
    golden = _load()[str(seed)]
    got = _canon(run_seed(seed))
    for i, (g, s) in enumerate(zip(golden['steps'], got['steps'])):
        assert s['op'] == g['op'], f'seed={seed} step={i}: 操作列が記録と異なる（その前の手で振る舞いが変わった）'
        diff = [k for k in g['h'] if g['h'][k] != s['h'][k]]
        assert not diff, f'seed={seed} step={i} op={g["op"]}: 観測値が記録と異なる {diff}'
    assert len(got['steps']) == len(golden['steps'])
    assert got['final'] == golden['final'], f'seed={seed}: 最後の状態が記録と異なる'


def test_golden_record_covers_every_operation_kind():
    """守るもの: 記録が、操作の種類・状態をひととおり含んでいる（空の記録で素通りしない）。"""
    golden = _load()
    kinds = set()
    for rec in golden.values():
        for s in rec['steps']:
            op = s['op']
            kinds.add(op[2] if op[0] == 'viewer' else op[0])
    assert {'grc', 'gre', 'cand', 'none', 'typed', 'typedtext', 'typedoff', 'editnone',
            'editnoneoff', 'adopt', 'page_cand', 'page_none'} <= kinds
    states = {r['state'] for rec in golden.values() for f in rec['final']['viewer'].values()
              for r in f['regions']}
    assert {'pending', 'decided', 'unnamed', 'unnamed_off'} <= states


if __name__ == '__main__':
    if '--record' not in sys.argv:
        sys.exit('usage: test_region_selection_golden.py --record')
    out = {str(seed): _canon(run_seed(seed)) for seed in SEEDS}
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    with open(GOLDEN_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    print('recorded', GOLDEN_PATH, os.path.getsize(GOLDEN_PATH), 'bytes')
