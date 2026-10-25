
// 抽出結果プレビューの「図面を表示」viewer（表示専用）。選んだ区分のラベルを、元の文字の色を変えて示す。
// pan/zoom・全体表示・全画面は region_confirm_viewer.js から複製（共有しない。向こうはゴールデンテストの対象）。
export default function(component) {
  const { data, parentElement } = component;
  const NS = 'http://www.w3.org/2000/svg';
  let S = parentElement.__lv;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

  if (!S) {
    S = parentElement.__lv = { sig: null, vb: null, vb0: null, drag: null };
    const wrap = parentElement.querySelector('#lv-wrap');
    wrap.innerHTML =
      '<div class="lv-side" id="lv-side"></div>' +
      '<div class="lv-stage"><svg class="lv-cv" id="lv-cv" xmlns="' + NS + '" preserveAspectRatio="xMidYMid meet">' +
      '<style id="lv-hl"></style><rect id="lv-bg" fill="#000"/><g id="lv-base"></g></svg>' +
      '<div class="lv-note">線幅は拡大しても一定です</div></div>';
    S.wrap = wrap;
    S.svg = parentElement.querySelector('#lv-cv');
    S.side = parentElement.querySelector('#lv-side');
    const svg = S.svg;
    S.apply = () => { svg.setAttribute('viewBox', S.vb.join(' ')); };
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
      S.vb = [p.x - (p.x - S.vb[0]) * k, p.y - (p.y - S.vb[1]) * k, nw, nh]; S.apply();
    }, { passive: false });
    svg.addEventListener('pointerdown', (e) => {
      S.drag = { x: e.clientX, y: e.clientY, vb: S.vb.slice(), s: pt(e).s };
      svg.setPointerCapture(e.pointerId); svg.classList.add('drag');
    });
    svg.addEventListener('pointermove', (e) => {
      if (!S.drag) return;
      S.vb = [S.drag.vb[0] - (e.clientX - S.drag.x) * S.drag.s, S.drag.vb[1] - (e.clientY - S.drag.y) * S.drag.s, S.drag.vb[2], S.drag.vb[3]];
      S.apply();
    });
    const end = () => { S.drag = null; svg.classList.remove('drag'); };
    svg.addEventListener('pointerup', end); svg.addEventListener('pointercancel', end);
    svg.addEventListener('dblclick', () => { S.vb = S.vb0.slice(); S.apply(); });
    document.addEventListener('fullscreenchange', () => { if (S.render) S.render(); });
  }

  S.data = data;
  // 外枠の高さを明示する（v2 部品の中では親の高さが確定せず、100% が効かないため）
  if (data.height) S.wrap.style.height = data.height + 'px';

  const render = () => {
    const d = S.data;
    if (d.empty) { S.wrap.innerHTML = '<div style="padding:16px;color:#aaa">描画対象のエンティティがありません</div>'; return; }
    if (S.sig !== d.sig) {
      S.sig = d.sig;
      S.vb0 = [0, 0, d.viewBox[0], d.viewBox[1]]; S.vb = S.vb0.slice();
      S.svg.querySelector('#lv-bg').setAttribute('width', d.viewBox[0]);
      S.svg.querySelector('#lv-bg').setAttribute('height', d.viewBox[1]);
      S.svg.querySelector('#lv-base').innerHTML = d.svg;
    }
    // 元の文字そのものの色を変える（SVG の文字はラベルごとの CSS クラスを持つ）。選択を変えても描き直さない
    const rules = [];
    d.items.forEach((it) => {
      const col = d.colors[it.m];
      it.c.forEach((c) => {
        rules.push('#lv-cv .' + c[0] + '{' + c[1] + ':' + col + ';' +
          (c[1] === 'fill' ? 'stroke:' + col + ';stroke-width:0.6px;vector-effect:non-scaling-stroke' : '') + '}');
      });
    });
    S.svg.querySelector('#lv-hl').textContent = rules.join('\n');
    S.apply();

    let h = '<h4>機器符号候補</h4>';
    d.groups.forEach((g) => {
      h += '<div class="lv-row' + (g.selected ? '' : ' off') + '"><span class="lv-sw" style="background:' +
           (g.selected ? g.color : 'transparent') + '"></span><span>' + esc(g.mark) +
           (g.selected ? '' : '（選択なし）') + '</span><span class="lv-cnt">' + g.count + ' 件</span></div>';
    });
    if (d.unplaced > 0) {
      h += '<div class="lv-warn">図面に描かれていないラベルが ' + d.unplaced + ' 件あります（色を付けられません）。</div>';
    }
    h += '<h4>操作</h4><button data-act="fit">全体表示</button><button data-act="fs">' +
         (document.fullscreenElement ? '全画面を終了' : '全画面') + '</button><div id="lv-fsnote" style="color:#e8a"></div>' +
         '<div style="color:#888;margin-top:6px;line-height:1.5">ホイール: 拡大縮小<br>ドラッグ: 移動<br>ダブルクリック: 全体表示</div>' +
         (d.hasFont ? '' : '<div class="lv-warn" style="margin-top:8px">日本語フォントが見つからないため、日本語の文字は表示されません。</div>');
    S.side.innerHTML = h;
    S.side.querySelectorAll('button[data-act]').forEach((b) => b.addEventListener('click', () => {
      const act = b.getAttribute('data-act');
      if (act === 'fit') { S.vb = S.vb0.slice(); S.apply(); return; }
      const note = S.side.querySelector('#lv-fsnote'); note.textContent = '';
      if (document.fullscreenElement) { document.exitFullscreen(); return; }
      const p = S.wrap.requestFullscreen ? S.wrap.requestFullscreen() : null;
      if (!p) { note.textContent = 'この環境では全画面にできません'; return; }
      p.catch(() => { note.textContent = 'この環境では全画面にできません'; });
    }));
  };
  S.render = render;
  render();
}
