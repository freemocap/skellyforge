// Interactive presentation of the assembly schematic. No solver calculations here.
const ceresMap = {
  nodes: [], edges: [], width: 1, height: 1, key: '',
  selected: null, hovered: null, scale: 1, x: 0, y: 0,
  autoFit: true, installed: false, drag: null,
};
const mapKinds = {
  chest_line:{color:"#f6e58d",tag:"LANDMARK LINE / PREFERENCE"},
  length:{color:"#5fe6bc",tag:"AXIAL LENGTH / PARAMETER"},
  length_prior:{color:"#e6b6ad",tag:"LENGTH PRIOR / RESIDUAL"},
  length_motion:{color:"#7ac9ad",tag:"LENGTH MOTION / RESIDUAL"},
  pose_prior: {color:"#dba8ee",tag:"RELATIVE POSE / RESIDUAL"},
  displacement: {color: "#f2d17b", tag:"DISPLACEMENT / PARAMETER"},
  prior: {color:"#ed89cf",tag:"DISPLACEMENT PRIOR / RESIDUAL"},
  displacement_motion: {color:"#dfb35a",tag:"DISPLACEMENT MOTION / RESIDUAL"},
  quaternion: {color: '#bc9aff', tag: 'QUATERNION / PARAMETER'},
  position: {color: '#60d5e5', tag: 'POSITION / PARAMETER'},
  chain: {color: '#ffba6b', tag: 'CHAIN LANDMARK / RESIDUAL'},
  landmark: {color: '#ff849a', tag: 'LANDMARK / RESIDUAL'},
  angular: {color: '#94d47a', tag: 'ANGULAR MOTION / RESIDUAL'},
  linear: {color: '#6fbcff', tag: 'POSITION MOTION / RESIDUAL'},
};
const mapEscape = text => String(text).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const hiddenMapKinds=new Set();
function renderCeresMap({parameters, residuals, indices}) {
  ceresMap.input={parameters,residuals,indices};
  const region=el('body-region').value;
  const focus=el('body-focus').value;
  if(region!=='all'||focus!=='all'){
    const included=b=>focus!=='all'?b===Number(focus):experiment.bodies[b]?.region===region;
    residuals=residuals.filter(r=>included(r.bodyIndex));
    const dependencies=new Set(residuals.flatMap(r=>r.connections));
    parameters=parameters.filter(p=>included(p.bodyIndex)||dependencies.has(p.id));
  }
  parameters=parameters.filter(p=>!hiddenMapKinds.has(p.kind||(p.quaternion?'quaternion':'position')));
  residuals=residuals.filter(r=>!hiddenMapKinds.has(r.kind));
  el('map-scope').textContent=`Viewing ${focus==='all'?region:experiment.bodies[Number(focus)].label}: ${parameters.length} parameter blocks and ${residuals.length} residual groups in this time window. Upstream parameter dependencies are retained. Filters change the display only.`;
  const map = ceresMap;
  const key = experiment.id + '|' + el('mode').value+'|'+region+'|'+focus;
  if (map.key !== key) {
    map.key = key; map.selected = null; map.hovered = null; map.autoFit = true;
  }
  const columnWidth = 570, cardHeight = 74, rowHeight = 94;
  const rows = Math.max(...indices.map(index => Math.max(parameters.filter(p => p.index === index).length,residuals.filter(r=>r.index===index).length)));
  const temporalY = 92 + rows * rowHeight;
  map.width = Math.max(570, indices.length * columnWidth);
  const temporalColumns=Math.max(1,Math.floor(map.width/280));
  map.height = temporalY + (residuals.some(r => r.index === null) ? 42+94*Math.ceil(residuals.filter(r=>r.index===null).length/temporalColumns) : 20);
  map.nodes = [];
  let backgrounds = '';
  indices.forEach((index, column) => {
    const x = column * columnWidth;
    backgrounds += `<rect x="${x+5}" y="5" width="550" height="${temporalY-15}" rx="12" fill="#172536" stroke="${index===frameIndex?'#65dae5':'#32465d'}"/><text x="${x+20}" y="32" fill="#d4e5f5" font-size="19">Frame ${index} · ${selectedSequence.times[index].toFixed(2)} s${index===frameIndex?' · selected':''}</text><text x="${x+20}" y="53" fill="#9eafc2" font-size="11">PARAMETER BLOCKS</text><text x="${x+300}" y="53" fill="#9eafc2" font-size="11">FRAME RESIDUAL BLOCKS</text>`;
    parameters.filter(p => p.index === index).forEach((p, row) => {
      map.nodes.push({...p, kind: p.kind||(p.quaternion?'quaternion':'position'),
        x:x+18, y:68+row*rowHeight, w:245, h:cardHeight,
        title:p.label, subtitle:p.kind==='length'?`Scalar / mm; bounds ${p.lower.toFixed(1)} to ${p.upper.toFixed(1)}`:p.kind==='displacement'?(p.values.length===3?'Parent-local XYZ / mm; no hard bounds':`Scalar / mm; bounds +/-${p.bound}`):p.quaternion?'wxyz · QuaternionManifold':'XYZ / mm · Euclidean',
        detail:`Frame ${p.index}; ${p.values.length===1?'1 stored value / 1 tangent dimension':p.quaternion?'4 stored values / 3 tangent dimensions':'3 stored values / 3 tangent dimensions'}. Values: ${p.values.map(v=>v.toFixed(5)).join(', ')}`});
    });
    residuals.filter(r => r.index === index).forEach((r, row) => {
      map.nodes.push({...r, id:`residual-${r.kind}-${r.connections.join("_")}`,
        x:x+295, y:68+row*rowHeight, w:245, h:cardHeight,
        title:r.bodyLabel, subtitle:r.name,
        detail:r.detail+'\n'+r.formula});
    });
  });
  const temporal = residuals.filter(r => r.index === null);
  if (temporal.length) backgrounds += `<text x="20" y="${temporalY+14}" fill="#cfdfed" font-size="18">TEMPORAL RESIDUAL BLOCKS · connect parameter blocks across frames</text>`;
  temporal.forEach((r, i) => {
    const cellWidth = map.width / temporalColumns;
    map.nodes.push({...r, id:`residual-${r.kind}-${r.connections.join("_")}`,
      x:(i%temporalColumns)*cellWidth+18, y:temporalY+30+Math.floor(i/temporalColumns)*94, w:cellWidth-36, h:cardHeight,
      title:r.bodyLabel, subtitle:r.name, detail:r.detail+'\n'+r.formula});
  });
  const byId = new Map(map.nodes.map(n => [n.id,n]));
  map.edges = map.nodes.filter(n => n.connections).flatMap(n => n.connections.filter(source=>byId.has(source)).map(source => ({source,target:n.id})));
  if (!byId.has(map.selected)) map.selected = null;
  if (!byId.has(map.hovered)) map.hovered = null;
  let edges = '';
  for (const edge of map.edges) {
    const a=byId.get(edge.source), b=byId.get(edge.target);
    let path;
    if (b.index === null) {
      const ax=a.x+a.w/2, ay=a.y+a.h, bx=b.x+b.w/2, by=b.y;
      path=`M ${ax} ${ay} C ${ax} ${by-24}, ${bx} ${by-24}, ${bx} ${by}`;
    } else {
      const ax=a.x+a.w, ay=a.y+a.h/2, bx=b.x, by=b.y+b.h/2;
      path=`M ${ax} ${ay} C ${ax+18} ${ay}, ${bx-18} ${by}, ${bx} ${by}`;
    }
    edges += `<path data-source="${edge.source}" data-target="${edge.target}" d="${path}" fill="none" stroke="${mapKinds[b.kind].color}" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
  }
  let cards = '';
  for (const node of map.nodes) {
    const kind=mapKinds[node.kind];
    cards += `<g data-node="${node.id}" tabindex="0" role="button" aria-label="${mapEscape(node.title+'; '+node.subtitle)}" style="cursor:pointer"><title>${mapEscape(node.detail)}</title><rect x="${node.x}" y="${node.y}" width="${node.w}" height="${node.h}" rx="8" fill="#1b2a3d" stroke="${kind.color}" stroke-width="1.5"/><rect x="${node.x}" y="${node.y}" width="4" height="${node.h}" rx="2" fill="${kind.color}"/><text x="${node.x+12}" y="${node.y+17}" fill="${kind.color}" font-size="9">${kind.tag}</text><text x="${node.x+12}" y="${node.y+38}" fill="#f0f6ff" font-size="14">${mapEscape(node.title)}</text><text x="${node.x+12}" y="${node.y+59}" fill="#afc2d8" font-size="11">${mapEscape(node.subtitle)}</text></g>`;
  }
  el('problem-series').innerHTML = `<svg width="100%" height="100%" aria-label="Interactive Ceres dependency map"><g data-map-content>${backgrounds}<g class="map-edges">${edges}</g>${cards}</g></svg>`;
  installCeresMapControls();
  if (map.autoFit) fitCeresMap(); else transformCeresMap();
  highlightCeresMap();
}
function transformCeresMap() {
  const map=ceresMap;
  el('problem-series').querySelector('[data-map-content]').setAttribute('transform',`translate(${map.x} ${map.y}) scale(${map.scale})`);
  el('map-zoom').textContent=Math.round(map.scale*100)+'%';
}
function fitCeresMap() {
  const host=el('problem-series'),map=ceresMap;
  const width=host.clientWidth,height=host.clientHeight;
  map.scale=Math.max(.05,Math.min((width-24)/map.width,(height-24)/map.height,1.5));
  map.x=(width-map.width*map.scale)/2;map.y=(height-map.height*map.scale)/2;
  map.autoFit=true;transformCeresMap();
}
function zoomCeresMap(factor,x,y) {
  const host=el('problem-series'),map=ceresMap;
  x??=host.clientWidth/2;y??=host.clientHeight/2;
  const scale=Math.max(.05,Math.min(4,map.scale*factor)),ratio=scale/map.scale;
  map.x=x-(x-map.x)*ratio;map.y=y-(y-map.y)*ratio;map.scale=scale;
  map.autoFit=false;transformCeresMap();
}
function highlightCeresMap() {
  const map=ceresMap,id=map.hovered||map.selected;
  const related=new Set(id?[id]:[]);
  for (const edge of map.edges) if (edge.source===id||edge.target===id) {related.add(edge.source);related.add(edge.target);}
  for (const node of el('problem-series').querySelectorAll('[data-node]')) {
    node.style.opacity=!id||related.has(node.dataset.node)?'1':'.18';
    node.setAttribute('aria-pressed',String(node.dataset.node===map.selected));
  }
  for (const edge of el('problem-series').querySelectorAll('[data-source]')) {
    const active=edge.dataset.source===id||edge.dataset.target===id;
    edge.style.opacity=active?'1':el('map-all').checked?(id?'.06':'.18'):'0';
    edge.style.strokeWidth=active?'3':'1.2';
  }
  const node=map.nodes.find(n=>n.id===id);
  el('problem-selection').textContent=node?
    `${map.selected===id?'PINNED':'HOVER'} · ${node.title} · ${node.subtitle}\n${node.detail}\n${node.connections?'Uses parameter blocks: ':'Used by residual groups: '}${node.connections?node.connections.join(', '):map.edges.filter(e=>e.source===id).map(e=>map.nodes.find(n=>n.id===e.target).title+' / '+map.nodes.find(n=>n.id===e.target).subtitle).join('; ')}`:
    'Hover a block to trace its direct connections. Click to pin. Drag background to pan; wheel to zoom. Double-click a block to zoom to it. Escape clears selection.';
}
function installCeresMapControls() {
  if (ceresMap.installed) return;
  ceresMap.installed=true;
  const host=el('problem-series'),map=ceresMap;
  const refresh=()=>{map.autoFit=true;renderCeresMap(map.input);};
  for(const [kind,style] of Object.entries(mapKinds)){
    const label=document.createElement('label');label.style.color=style.color;
    const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.checked=true;checkbox.id='map-type-'+kind;
    checkbox.addEventListener('change',()=>{if(checkbox.checked)hiddenMapKinds.delete(kind);else hiddenMapKinds.add(kind);refresh();});
    label.appendChild(checkbox);const text=document.createElement('span');text.textContent=style.tag;label.appendChild(text);el('map-filters').appendChild(label);
  }
  const preset=predicate=>{for(const kind of Object.keys(mapKinds)){const show=predicate(kind);el('map-type-'+kind).checked=show;if(show)hiddenMapKinds.delete(kind);else hiddenMapKinds.add(kind);}refresh();};
  el('map-types-all').onclick=()=>preset(()=>true);
  el('map-types-parameters').onclick=()=>preset(kind=>['quaternion','position','displacement','length'].includes(kind));
  el('map-types-motion').onclick=()=>preset(kind=>['angular','linear','displacement_motion','length_motion'].includes(kind));
  const nodeId=event=>event.target.closest('[data-node]')?.dataset.node;
  el('map-fit').onclick=fitCeresMap;
  el('map-zoom-in').onclick=()=>zoomCeresMap(1.3);
  el('map-zoom-out').onclick=()=>zoomCeresMap(1/1.3);
  el('map-clear').onclick=()=>{map.selected=null;map.hovered=null;highlightCeresMap();};
  el('map-all').addEventListener('change',highlightCeresMap);
  host.addEventListener('wheel',event=>{event.preventDefault();const bounds=host.getBoundingClientRect();zoomCeresMap(Math.exp(-event.deltaY*.0015),event.clientX-bounds.left,event.clientY-bounds.top);},{passive:false});
  host.addEventListener('pointermove',event=>{
    if (map.drag) {map.x=map.drag.x+event.clientX-map.drag.pointerX;map.y=map.drag.y+event.clientY-map.drag.pointerY;map.autoFit=false;transformCeresMap();return;}
    const id=nodeId(event)||null;if (id!==map.hovered){map.hovered=id;highlightCeresMap();}
  });
  host.addEventListener('pointerleave',()=>{map.hovered=null;highlightCeresMap();});
  host.addEventListener('pointerdown',event=>{if(event.button!==0||nodeId(event))return;map.drag={x:map.x,y:map.y,pointerX:event.clientX,pointerY:event.clientY};host.setPointerCapture(event.pointerId);event.preventDefault();});
  const endDrag=event=>{map.drag=null;if(host.hasPointerCapture(event.pointerId))host.releasePointerCapture(event.pointerId);};
  host.addEventListener('pointerup',endDrag);host.addEventListener('pointercancel',endDrag);host.addEventListener('lostpointercapture',()=>map.drag=null);
  host.addEventListener('click',event=>{const id=nodeId(event);if(id){map.selected=map.selected===id?null:id;map.hovered=null;highlightCeresMap();}});
  host.addEventListener('dblclick',event=>{const node=map.nodes.find(n=>n.id===nodeId(event));if(!node)return;map.scale=1.5;map.x=host.clientWidth/2-(node.x+node.w/2)*map.scale;map.y=host.clientHeight/2-(node.y+node.h/2)*map.scale;map.autoFit=false;map.selected=node.id;map.hovered=null;transformCeresMap();highlightCeresMap();});
  host.addEventListener('focusin',event=>{map.hovered=nodeId(event)||null;highlightCeresMap();});
  host.addEventListener('keydown',event=>{
    const id=nodeId(event);
    if((event.key==='Enter'||event.key===' ')&&id){event.preventDefault();map.selected=id;map.hovered=null;highlightCeresMap();}
    else if(event.key==='Escape'){map.selected=null;map.hovered=null;highlightCeresMap();}
    else if(event.key==='+'||event.key==='='){event.preventDefault();zoomCeresMap(1.3);}
    else if(event.key==='-'){event.preventDefault();zoomCeresMap(1/1.3);}
    else if(event.key.toLowerCase()==='f'){event.preventDefault();fitCeresMap();}
  });
  new ResizeObserver(()=>{if(map.nodes.length&&map.autoFit)fitCeresMap();}).observe(host);
}
