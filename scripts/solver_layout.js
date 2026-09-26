// Shared sidebar splitter; independent of experiment geometry.
function setupSidebar(){
const divider=el('divider');let panelWidth=420;
function resizePanel(width){
  const maximum=Math.max(120,innerWidth-129),minimum=Math.min(280,maximum);
  panelWidth=Math.max(minimum,Math.min(maximum,width));
  document.documentElement.style.setProperty('--panel-width',`${panelWidth}px`);
  divider.setAttribute('aria-valuemin',minimum);divider.setAttribute('aria-valuemax',maximum);
  divider.setAttribute('aria-valuenow',Math.round(panelWidth));
  divider.setAttribute('aria-valuetext',`Controls panel ${Math.round(panelWidth)} pixels wide`);
}
divider.addEventListener('pointerdown',event=>{if(event.button!==0)return;divider.setPointerCapture(event.pointerId);divider.classList.add('dragging');document.body.classList.add('resizing');event.preventDefault();});
divider.addEventListener('pointermove',event=>{if(divider.hasPointerCapture(event.pointerId))resizePanel(innerWidth-event.clientX-4);});
function endResize(){divider.classList.remove('dragging');document.body.classList.remove('resizing');}
divider.addEventListener('pointerup',event=>{if(divider.hasPointerCapture(event.pointerId))divider.releasePointerCapture(event.pointerId);endResize();});
divider.addEventListener('pointercancel',endResize);divider.addEventListener('lostpointercapture',endResize);
divider.addEventListener('dblclick',()=>resizePanel(420));
divider.addEventListener('keydown',event=>{if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();resizePanel(panelWidth+(event.key==='ArrowLeft'?1:-1)*(event.shiftKey?50:20));}else if(event.key==='Home'){event.preventDefault();resizePanel(420);}});
window.addEventListener('resize',()=>resizePanel(panelWidth));resizePanel(panelWidth);

}
