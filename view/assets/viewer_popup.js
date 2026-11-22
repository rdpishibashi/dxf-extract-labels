(function(){
// zoom-sens:begin  ズーム感度（回転量に比例）。テストがこの範囲を Node で実行する
var ZOOM_SENS_KEY = 'dxfViewerZoomSensitivity', ZOOM_SENS_MIN = 0.1, ZOOM_SENS_MAX = 3, ZOOM_SENS_DEFAULT = 1;
var zoomSensMem = ZOOM_SENS_DEFAULT;
function clampZoomSens(v) {
  v = Number(v);
  if (!isFinite(v)) return ZOOM_SENS_DEFAULT;
  return Math.round(Math.min(ZOOM_SENS_MAX, Math.max(ZOOM_SENS_MIN, v)) * 10) / 10;
}
function getZoomSens() {
  try { var s = localStorage.getItem(ZOOM_SENS_KEY); if (s !== null) return clampZoomSens(s); } catch (err) {}
  return zoomSensMem;
}
function setZoomSens(v) {
  zoomSensMem = clampZoomSens(v);
  try { localStorage.setItem(ZOOM_SENS_KEY, String(zoomSensMem)); } catch (err) {}
  return zoomSensMem;
}
// 戻り値は viewBox の幅に掛ける倍率（>1 で縮小表示、<1 で拡大表示）
function zoomFactor(e, sens) {
  var d = e.deltaY;
  if (e.deltaMode === 1) d *= 40; else if (e.deltaMode === 2) d *= 800;  // 行・ページ単位をピクセルに換算
  var rate = e.ctrlKey ? 0.01 : 0.002;  // ピンチ（ctrlKey 付き）は1回の回転量が小さいので係数を大きくする
  var x = Math.max(-0.5, Math.min(0.5, d * rate * sens));  // 1回で跳びすぎないよう上限
  return Math.exp(x);
}
// zoom-sens:end
  var svg=document.querySelector('svg[data-popup-cv]');
  var vb0=svg.getAttribute('viewBox').split(' ').map(Number);
  var vb=vb0.slice();
  function apply(){
    svg.setAttribute('viewBox',vb.join(' '));
    var r=svg.getBoundingClientRect();
    if(r.width&&r.height)svg.style.setProperty('--fs',(Math.max(vb[2]/r.width,vb[3]/r.height)*13)+'px');}
  function reset(){vb=vb0.slice();apply();}
  function pt(e){var r=svg.getBoundingClientRect();
    var s=Math.max(vb[2]/r.width,vb[3]/r.height);
    var offx=(r.width-vb[2]/s)/2, offy=(r.height-vb[3]/s)/2;
    return {x:vb[0]+(e.clientX-r.left-offx)*s,y:vb[1]+(e.clientY-r.top-offy)*s,s:s};}
  svg.addEventListener('wheel',function(e){
    e.preventDefault();
    var p=pt(e),k=zoomFactor(e,getZoomSens());
    var nw=vb[2]*k,nh=vb[3]*k;
    if(nw>vb0[2]*20||nw<vb0[2]/2000)return;
    vb=[p.x-(p.x-vb[0])*k,p.y-(p.y-vb[1])*k,nw,nh];apply();
  },{passive:false});
  var drag=null;
  svg.addEventListener('pointerdown',function(e){
    drag={x:e.clientX,y:e.clientY,vb:vb.slice(),s:pt(e).s};
    svg.setPointerCapture(e.pointerId);svg.classList.add('drag');});
  svg.addEventListener('pointermove',function(e){
    if(!drag)return;
    vb=[drag.vb[0]-(e.clientX-drag.x)*drag.s,drag.vb[1]-(e.clientY-drag.y)*drag.s,drag.vb[2],drag.vb[3]];apply();});
  function end(){drag=null;svg.classList.remove('drag');}
  svg.addEventListener('pointerup',end);svg.addEventListener('pointercancel',end);
  svg.addEventListener('dblclick',reset);
  document.getElementById('fit').addEventListener('click',reset);
  var zs=document.querySelector('[data-zoom-sens]'),zv=document.querySelector('[data-zoom-sens-val]');
  if(zs){
    var zsShow=function(v){zs.value=v;if(zv)zv.textContent='\u00d7'+Number(v).toFixed(1);};
    zsShow(getZoomSens());
    zs.addEventListener('input',function(){zsShow(setZoomSens(zs.value));});
  }
  window.addEventListener('resize',apply);
  apply();
})();
