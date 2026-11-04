(function(){
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
    var p=pt(e),k=e.deltaY<0?0.8:1.25;
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
  window.addEventListener('resize',apply);
  apply();
})();
