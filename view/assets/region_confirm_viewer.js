
export default function(component) {
  const { data, parentElement, setTriggerValue } = component;
  const NS = 'http://www.w3.org/2000/svg';
  let S = parentElement.__rp;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

  if (!S) {
    S = parentElement.__rp = { counter: 0, sig: null, vb: null, vb0: null, hidden: {}, drag: null };
    const wrap = parentElement.querySelector('#rp-wrap');
    wrap.innerHTML =
      '<div class="rp-side" id="rp-side"></div>' +
      '<div class="rp-stage"><svg class="rp-cv" id="rp-cv" xmlns="' + NS + '" preserveAspectRatio="xMidYMid meet">' +
      '<style id="rp-hl"></style><rect id="rp-bg" fill="#000"/><g id="rp-base"></g><g id="rp-ov"></g></svg>' +
      '<div class="rp-note">線幅は拡大しても一定です</div></div>';
    S.wrap = wrap;
    S.svg = parentElement.querySelector('#rp-cv');
    S.side = parentElement.querySelector('#rp-side');
    const svg = S.svg;
    const apply = () => {
      svg.setAttribute('viewBox', S.vb.join(' '));
      const r = svg.getBoundingClientRect();
      if (r.width && r.height)
        svg.style.setProperty('--fs', (Math.max(S.vb[2] / r.width, S.vb[3] / r.height) * 13) + 'px');
    };
    S.apply = apply;
    const pt = (e) => {
      const r = svg.getBoundingClientRect();
      const s = Math.max(S.vb[2] / r.width, S.vb[3] / r.height);
      const offx = (r.width - S.vb[2] / s) / 2, offy = (r.height - S.vb[3] / s) / 2;
      return { x: S.vb[0] + (e.clientX - r.left - offx) * s, y: S.vb[1] + (e.clientY - r.top - offy) * s, s };
    };
    svg.addEventListener('wheel', (e) => {
      e.preventDefault();
      const p = pt(e), k = e.deltaY < 0 ? 0.8 : 1.25;
      const nw = S.vb[2] * k, nh = S.vb[3] * k;
      if (nw > S.vb0[2] * 20 || nw < S.vb0[2] / 2000) return;
      S.vb = [p.x - (p.x - S.vb[0]) * k, p.y - (p.y - S.vb[1]) * k, nw, nh]; apply();
    }, { passive: false });
    svg.addEventListener('pointerdown', (e) => {
      S.drag = { x: e.clientX, y: e.clientY, vb: S.vb.slice(), s: pt(e).s };
      svg.setPointerCapture(e.pointerId); svg.classList.add('drag');
    });
    svg.addEventListener('pointermove', (e) => {
      if (!S.drag) return;
      S.vb = [S.drag.vb[0] - (e.clientX - S.drag.x) * S.drag.s, S.drag.vb[1] - (e.clientY - S.drag.y) * S.drag.s, S.drag.vb[2], S.drag.vb[3]];
      apply();
    });
    const end = () => { S.drag = null; svg.classList.remove('drag'); };
    svg.addEventListener('pointerup', end); svg.addEventListener('pointercancel', end);
    svg.addEventListener('dblclick', () => { S.vb = S.vb0.slice(); apply(); });
    document.addEventListener('fullscreenchange', () => { if (S.render) S.render(); });
  }

  S.data = data;
  // 外枠の高さを明示する（v2 部品の中では親の高さが確定せず、100% が効かないため）
  if (data.height) S.wrap.style.height = data.height + 'px';
  S.setTrigger = setTriggerValue;

  const send = (payload) => {
    S.counter += 1;
    S.setTrigger('choice', Object.assign({ seq: Date.now() * 1000 + S.counter }, payload));
  };

  const choose = (rid, kind, idx, text, typedText) => {
    const d = S.data;
    if (d.locked) return;
    const ch = d.choices[String(rid)];
    if (!ch) return;
    // その場で見た目を更新（Python 側の再実行を待たない）
    if (kind === 'cand') {
      if (ch.cands.some((c) => c.idx === idx && c.selected)) return;
      ch.cands.forEach((c) => { c.selected = (c.idx === idx); }); ch.none = false; ch.typed = false;
    } else if (kind === 'none') {
      if (ch.none) return;
      ch.cands.forEach((c) => { c.selected = false; }); ch.none = true; ch.typed = false;
    } else if (kind === 'typed') {
      if (ch.typed) return;
      ch.cands.forEach((c) => { c.selected = false; }); ch.none = false; ch.typed = true; ch.typedMatch = matchName(ch.typedtext);
    } else if (kind === 'typedoff') {
      if (!ch.typed && !ch.typedtext) return;
      ch.typed = false; ch.typedtext = '';
    } else if (kind === 'editnone') {
      if (ch.none) return;
      ch.none = true;
    } else if (kind === 'editnoneoff') {
      if (!ch.none) return;
      ch.none = false;
    } else if (kind === 'typedtext') {
      ch.cands.forEach((c) => { c.selected = false; }); ch.none = false; ch.typed = true; ch.typedtext = typedText; ch.typedMatch = matchName(typedText);
    }
    send({ region: rid, kind, idx: idx === undefined ? null : idx, text: text || '', typed_text: typedText === undefined ? null : typedText });
    S.render();
  };

  const norm = (t) => String(t || '').normalize('NFKC').trim().toUpperCase();
  const matchName = (t) => { const k = norm(t); return k ? ((S.data.labelNames || []).find((n) => norm(n) === k) || null) : null; };
  const isHidden = (r) => {
    if (r.state === 'pending') { const c = S.data.choices[String(r.id)]; return !!(c && c.none); }  // 「領域選択しない」なら図面に出さない
    return !!S.hidden[r.key];
  };
  const regColor = (r, C) => r.state === 'decided' || r.state === 'unnamed' ? C.decided : (r.state === 'unnamed_off' ? C.off : C.pending);
  const geomText = (g, col) => {
    const size = g.size;
    const style = size ? 'font-size:' + size.toFixed(2) + 'px;stroke-width:' + (size * 0.25).toFixed(2) + 'px' : '';
    const rot = g.rot ? ' transform="rotate(' + g.rot.toFixed(2) + ' ' + g.x.toFixed(2) + ' ' + g.y.toFixed(2) + ')"' : '';
    return '<text x="' + g.x.toFixed(2) + '" y="' + g.y.toFixed(2) + '" fill="' + col + '" text-anchor="' + g.anchor +
           '" dominant-baseline="' + g.base + '"' + rot + ' style="' + style + '">' + esc(g.text) + '</text>';
  };

  // 元の文字そのものの色を変える（SVG の文字は専用の CSS クラスを持つ）。クラスが無いラベルだけ、重ね書きにする
  let hlRules = [];
  const hlLabel = (g, col) => {
    if (!g.cls || !g.cls.length) return geomText(g, col);
    g.cls.forEach((c) => {
      hlRules.push('#rp-cv .' + c[0] + '{' + c[1] + ':' + col + ';' +
        (c[1] === 'fill' ? 'stroke:' + col + ';stroke-width:0.6px;vector-effect:non-scaling-stroke' : '') + '}');
    });
    return '';
  };

  const render = () => {
    const d = S.data, C = d.colors;
    hlRules = [];
    if (d.empty) { S.wrap.innerHTML = '<div style="padding:16px;color:#aaa">描画対象のエンティティがありません</div>'; return; }
    if (S.sig !== d.sig) {
      S.sig = d.sig;
      S.vb0 = [0, 0, d.viewBox[0], d.viewBox[1]]; S.vb = S.vb0.slice();
      S.svg.querySelector('#rp-bg').setAttribute('width', d.viewBox[0]);
      S.svg.querySelector('#rp-bg').setAttribute('height', d.viewBox[1]);
      S.svg.querySelector('#rp-base').innerHTML = d.svg;
    }
    // --- 図面上の重ね表示 ---
    let ov = '';
    d.regions.forEach((r) => {
      if (isHidden(r)) return;
      const col = regColor(r, C);
      ov += '<polygon points="' + r.points.map((p) => p[0].toFixed(2) + ',' + p[1].toFixed(2)).join(' ') +
            '" fill="' + col + '" stroke="' + col + '"/>';
      if (r.state !== 'pending') {
        // 確定した領域の名称は元のラベルの位置に（同じ文字列が複数あればすべて）。無名領域は多角形の左上
        if (r.geoms && r.geoms.length) r.geoms.forEach((g) => { ov += hlLabel(g, col); });
        else ov += '<text x="' + (r.anchor[0] + 4) + '" y="' + (r.anchor[1] + 16) + '" fill="' + col + '">' + esc(r.name) + '</text>';
      }
    });
    d.regions.forEach((r) => {
      if (r.state !== 'pending' || isHidden(r)) return;
      const ch = d.choices[String(r.id)];
      (ch ? ch.cands : []).forEach((c) => {
        const col = c.selected ? C.candSel : C.candOther;
        (c.geoms || []).forEach((g) => { ov += hlLabel(g, col); });
      });
    });
    S.svg.querySelector('#rp-ov').innerHTML = ov;
    S.svg.querySelector('#rp-hl').textContent = hlRules.join('\n');
    S.apply();

    // --- 左パネル ---
    const scroll = S.side.scrollTop;
    // 「領域」には確定した領域だけを出す（要選択の領域は「抽出領域の選択」の中で表示を切り替える）
    let h = '<h4>領域</h4>';
    const dis = d.locked ? ' dis' : '';
    const msg = (ch) => ch.typedMatch ? '<div class="rp-msg" style="color:#7fdc7f">採用: ' + esc(ch.typedMatch) + '</div>'
                                      : '<div class="rp-msg" style="color:#e8a">一致するラベルなし（未選択になります）</div>';
    d.regions.filter((r) => r.state !== 'pending').forEach((r) => {
      const col = regColor(r, C);
      const ch0 = d.choices[String(r.id)];
      const editable = !!(ch0 && r.editable);
      // 確定済み・無名の領域の行のチェックは「採用する」（外す＝「（領域選択しない）」と同じ）。それ以外（除外など）は図面への表示切り替え
      const on = editable ? !(ch0.none || (ch0.typed && !ch0.typedMatch)) : !S.hidden[r.key];
      h += '<div class="rp-reg"><label><input type="checkbox" ' + (editable ? 'data-adopt="' + r.id + '" ' + (d.locked ? 'disabled ' : '')
           : 'data-vis="' + r.key + '" ') + (on ? 'checked' : '') + '>' +
           '<span class="rp-sw" style="background:' + col + '"></span><span>' + esc(r.label) +
           '　' + esc(r.name) + (r.note ? '（' + esc(r.note) + '）' : '') + '</span></label>';
      const ch = d.choices[String(r.id)];
      if (ch && r.editable) {
        // 確定済み・無名の領域: 直下で名称を入力し直す／領域選択しない。入力欄を空にすると自動の名称に戻る
        h += '<h4>領域名を入力</h4>' +
             '<input type="text" data-ex="' + r.id + '" value="' + esc(ch.typedtext || '') + '" ' + (d.locked ? 'disabled ' : '') +
             'placeholder="図面枠内のラベルと完全一致（全角・小文字可）"><div class="rp-sg" data-sg="e' + r.id + '"></div>';
        if (ch.typed && !ch.none) h += msg(ch);
        h += '<label class="' + dis.trim() + '"><input type="checkbox" data-en="' + r.id + '" ' + (ch.none ? 'checked ' : '') +
             (d.locked ? 'disabled' : '') + '><span>（領域選択しない）</span></label>';
      }
      h += '</div>';
    });
    h += '<h4>抽出領域の選択</h4>';
    d.regions.filter((r) => r.state === 'pending').forEach((r) => {
      const ch = d.choices[String(r.id)];
      h += '<div class="rp-reg"><b>' + esc(r.label) + '</b>';
      ch.cands.forEach((c) => {
        h += '<label class="' + dis.trim() + '"><input type="checkbox" data-c="' + r.id + ':' + c.idx + '" ' +
             (c.selected ? 'checked ' : '') + (d.locked ? 'disabled' : '') + '><span class="rp-sw" style="background:' +
             (c.selected ? C.candSel : C.candOther) + '"></span><span>' + esc(c.text) + '（距離 ' + c.dist + '）</span></label>';
      });
      h += '<label class="' + dis.trim() + '"><input type="checkbox" data-n="' + r.id + '" ' + (ch.none ? 'checked ' : '') +
           (d.locked ? 'disabled' : '') + '><span>（領域選択しない）</span></label>';
      h += '<label class="' + dis.trim() + '"><input type="checkbox" data-t="' + r.id + '" ' + (ch.typed ? 'checked ' : '') +
           (d.locked ? 'disabled' : '') + '><span>領域名を入力</span></label>';
      h += '<input type="text" data-tx="' + r.id + '" value="' + esc(ch.typedtext || '') + '" ' + (d.locked ? 'disabled ' : '') +
           'placeholder="図面枠内のラベルと完全一致（全角・小文字可）"><div class="rp-sg" data-sg="t' + r.id + '"></div>';
      if (ch.typed) {
        h += ch.typedMatch ? '<div class="rp-msg" style="color:#7fdc7f">採用: ' + esc(ch.typedMatch) + '</div>'
                           : '<div class="rp-msg" style="color:#e8a">一致するラベルなし（未選択になります）</div>';
      }
      h += '</div>';
    });
    h += '<h4>操作</h4><button data-act="fit">全体表示</button><button data-act="fs">' +
         (document.fullscreenElement ? '全画面を終了' : '全画面') + '</button><div id="rp-fsnote" style="color:#e8a"></div>' +
         '<div style="color:#888;margin-top:6px;line-height:1.5">ホイール: 拡大縮小<br>ドラッグ: 移動<br>ダブルクリック: 全体表示</div>' +
         (d.hasFont ? '' : '<div style="color:#e8a;margin-top:8px">日本語フォントが見つからないため、日本語の文字は表示されません。</div>');
    S.side.innerHTML = h;
    S.side.scrollTop = scroll;

    S.side.querySelectorAll('input[data-vis]').forEach((el) => el.addEventListener('change', () => {
      S.hidden[el.getAttribute('data-vis')] = !el.checked; render();
    }));
    S.side.querySelectorAll('input[data-c]').forEach((el) => el.addEventListener('change', () => {
      const [rid, idx] = el.getAttribute('data-c').split(':').map(Number);
      const c = S.data.choices[String(rid)].cands.find((x) => x.idx === idx);
      if (!el.checked) { render(); return; }
      choose(rid, 'cand', idx, c.text);
    }));
    S.side.querySelectorAll('input[data-n]').forEach((el) => el.addEventListener('change', () => {
      if (!el.checked) { render(); return; }
      choose(Number(el.getAttribute('data-n')), 'none');
    }));
    S.side.querySelectorAll('input[data-t]').forEach((el) => el.addEventListener('change', () => {
      const rid = Number(el.getAttribute('data-t'));
      if (!el.checked) {
        // 確定した領域は「領域名を入力」を外すと元の名称に戻る。要選択の領域は他の選択肢を選ぶ
        const r = S.data.regions.find((x) => x.id === rid);
        if (r && r.editable) choose(rid, 'typedoff'); else render();
        return;
      }
      choose(rid, 'typed');
    }));
    const commitEx = (el) => {
      const rid = Number(el.getAttribute('data-ex'));
      if (el.value.trim() === '') { choose(rid, 'typedoff', undefined, '', ''); return; }
      choose(rid, 'typedtext', undefined, '', el.value);
    };
    const commitTx = (el) => choose(Number(el.getAttribute('data-tx')), 'typedtext', undefined, '', el.value);
    S.side.querySelectorAll('input[data-ex]').forEach((el) => el.addEventListener('change', () => commitEx(el)));
    S.side.querySelectorAll('input[data-tx]').forEach((el) => el.addEventListener('change', () => commitTx(el)));
    S.side.querySelectorAll('input[data-adopt]').forEach((el) => el.addEventListener('change', () => {
      const rid = Number(el.getAttribute('data-adopt'));
      const c = S.data.choices[String(rid)];
      if (!el.checked) { choose(rid, 'editnone'); return; }
      choose(rid, c.none ? 'editnoneoff' : 'typedoff');  // 入力名が不一致で未選択だった場合は自動の名称に戻す
    }));
    S.side.querySelectorAll('input[data-en]').forEach((el) => el.addEventListener('change', () => {
      const rid = Number(el.getAttribute('data-en'));
      choose(rid, el.checked ? 'editnone' : 'editnoneoff');
    }));
    // 入力補完: 全角・半角・大小文字を区別せず、含まれるラベルを候補に出す（クリックで入力）
    S.side.querySelectorAll('input[data-ex], input[data-tx]').forEach((el) => {
      const isEx = el.hasAttribute('data-ex');
      const box = S.side.querySelector('[data-sg="' + (isEx ? 'e' : 't') + el.getAttribute(isEx ? 'data-ex' : 'data-tx') + '"]');
      el.addEventListener('input', () => {
        const k = norm(el.value);
        const hit = k ? (S.data.labelNames || []).filter((n) => norm(n).includes(k)).slice(0, 8) : [];
        box.innerHTML = hit.map((n) => '<button type="button" data-pick="' + esc(n) + '">' + esc(n) + '</button>').join('');
        box.querySelectorAll('button').forEach((b) => b.addEventListener('mousedown', (ev) => {
          ev.preventDefault();
          el.value = b.getAttribute('data-pick'); box.innerHTML = '';
          if (isEx) commitEx(el); else commitTx(el);
        }));
      });
    });
    S.side.querySelectorAll('button[data-act]').forEach((b) => b.addEventListener('click', () => {
      const act = b.getAttribute('data-act');
      if (act === 'fit') { S.vb = S.vb0.slice(); S.apply(); return; }
      const note = S.side.querySelector('#rp-fsnote'); note.textContent = '';
      if (document.fullscreenElement) { document.exitFullscreen(); return; }
      const p = S.wrap.requestFullscreen ? S.wrap.requestFullscreen() : null;
      if (!p) { note.textContent = 'この環境では全画面にできません'; return; }
      p.catch(() => { note.textContent = 'この環境では全画面にできません'; });
    }));
  };
  S.render = render;
  render();
}
