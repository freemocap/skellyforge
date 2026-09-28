const captureDataset = typeof location === 'undefined' ? 'test' : (new URLSearchParams(location.search).get('dataset') || 'test');
if(captureDataset==='synthetic'){el('recording').src='/synthetic-fit';}
// Native properties are rendered directly. Only layout, filtering and text
// formatting live here; no residual equations or connectivity reconstruction.
Object.assign(mapKinds,{
  native_parameter:{color:'#60d5e5',tag:'PARAMETER'},
  native_manifold:{color:'#bc9aff',tag:'MANIFOLD PARAMETER'},
  native_constant:{color:'#a5b2c2',tag:'FIXED PARAMETER'},
  native_residual:{color:'#ffba6b',tag:'RESIDUAL'},
});
let nativeWindow=null,loadVersion=0;
const shortType=s=>s.replace(/class |struct /g,'').replace(/ceres::|skellyforge::/g,'');
function drawNativeProblem(){
  if(!nativeWindow)return;
  if(!nativeWindow.segment_names || nativeWindow.final.parameters.some(p=>p.frame===undefined)){el('summary').textContent='This capture predates named blocks. Regenerate it with poe solver-inspector-capture.';return;}
  const problem=nativeWindow[el('state').value],family=el('family').value,limit=Number(el('limit').value);
  const segment=el('segment').value;
  const visibleIds=new Set(problem.parameters.filter(p=>segment===''||p.segment===Number(segment)).map(p=>p.id));
  const matching=problem.residuals.filter(r=>(!family||r.purpose===family)&&r.parameter_ids.some(id=>visibleIds.has(id))).sort((a,b)=>b.cost-a.cost||a.id-b.id);
  const shown=limit?matching.slice(0,limit):matching;
  const linked=new Set(shown.flatMap(r=>r.parameter_ids));
  const parameters=problem.parameters.filter(p=>linked.has(p.id)&&visibleIds.has(p.id)).sort((a,b)=>a.frame-b.frame||a.segment-b.segment||a.quantity.localeCompare(b.quantity));
  const byId=new Map(problem.parameters.map(p=>[p.id,p]));
  const names=nativeWindow.segment_names||[];
  const name=p=>names[p.segment]||'Shared spine';
  const grouped=el('grouped').checked;
  const map=ceresMap;map.redraw=drawNativeProblem;
  const columns=nativeWindow.frame_end-nativeWindow.frame_start+1;
  map.width=columns*360;map.nodes=[];
  const rows=Array(columns).fill(0),parameterIds=new Map(),groups=new Map();
  for(const p of parameters){
    const key=grouped?`segment-${p.frame}-${p.segment}`:'p'+p.id;
    parameterIds.set(p.id,key);
    if(!groups.has(key))groups.set(key,[]);
    groups.get(key).push(p);
  }
  for(const [id,ps] of groups){
    const p=ps[0],frame=p.frame;
    map.nodes.push({id,index:frame,x:frame*360+15,y:65+rows[frame]++*78,w:330,h:65,
      kind:ps.every(p=>p.constant)?'native_constant':p.manifold_type?'native_manifold':'native_parameter',
      title:name(p).replaceAll('_',' '),
      subtitle:grouped?ps.map(p=>p.quantity.replace(' (mm)','').replace('World quaternion (wxyz)','Quaternion')).join(' / '):p.quantity,
      detail:`Frame ${frame+nativeWindow.frame_start} | ${name(p)}\n`+JSON.stringify(grouped?ps:p,null,2)});
  }
  const parameterHeight=Math.max(...rows)*78+100;
  rows.fill(0);const residualGroups=new Map();
  for(const r of shown){
    const frames=[...new Set(r.parameter_ids.map(id=>byId.get(id).frame))].sort((a,b)=>a-b);
    const key=grouped?r.purpose+'-'+frames.join('-'):'r'+r.id;
    if(!residualGroups.has(key))residualGroups.set(key,{rs:[],frames});
    residualGroups.get(key).rs.push(r);
  }
  for(const [key,{rs,frames}] of residualGroups){
    const frame=frames[Math.floor(frames.length/2)],r=rs[0];
    const connections=[...new Set(rs.flatMap(r=>r.parameter_ids.filter(id=>visibleIds.has(id)).map(id=>parameterIds.get(id))))];
    map.nodes.push({id:grouped?'group-'+key:key,index:frame,x:frame*360+15,y:parameterHeight+rows[frame]++*78,w:330,h:65,kind:'native_residual',
      title:r.purpose+(grouped?` (${rs.length})`:''),
      subtitle:`Frames ${frames.map(f=>f+nativeWindow.frame_start).join(', ')} | cost ${rs.reduce((v,r)=>v+r.cost,0).toPrecision(4)}`,
      detail:JSON.stringify(grouped?rs:r,null,2),connections});
  }
  map.height=parameterHeight+Math.max(...rows)*78+30;
  let backgrounds='';
  for(let f=0;f<columns;f++){
    const number=f+nativeWindow.frame_start,fixed=number<nativeWindow.active_start;
    backgrounds+=`<rect x="${f*360}" y="0" width="350" height="${map.height}" fill="${fixed?'#192330':'#142c39'}" rx="8"/><text x="${f*360+15}" y="25" fill="#e0e9f1" font-size="17">Frame ${number} - ${fixed?'Fixed history':'Active solve'}</text><text x="${f*360+15}" y="47" fill="#9cb7c8" font-size="12">Segment parameters above / residuals below</text>`;
  }

  const total=problem.residuals.reduce((sum,r)=>sum+r.cost,0);
  el('summary').textContent=`Frames ${nativeWindow.frame_start}–${nativeWindow.frame_end}; active from ${nativeWindow.active_start}. Actual totals: ${problem.parameters.length} parameters, ${problem.residuals.length} residual blocks; cost ${total.toPrecision(7)}. Showing ${shown.length}/${matching.length} matching residuals and their ${parameters.length} connected parameters.`;
  paintCeresMap(backgrounds);
}
async function loadNativeWindow(){
  const version=++loadVersion;
  try{
    const response=await fetch('.solver_inspections/'+captureDataset+'/'+el('window').value,{cache:'no-store'});
    if(!response.ok)throw Error(`Window load failed: ${response.status}`);
    const data=await response.json();if(version!==loadVersion)return;nativeWindow=data;
    const previousSegment=el('segment').value;
    el('segment').replaceChildren(new Option('All segments',''));
    data.segment_names.forEach((name,index)=>el('segment').add(new Option(name.replaceAll('_',' '),String(index))));
    el('segment').value=previousSegment||String(data.segment_names.indexOf('thoracic'));
    const selected=el('family').value;el('family').replaceChildren(new Option('All types',''));
    for(const type of [...new Set(data.final.residuals.map(r=>r.purpose))].sort())el('family').add(new Option(shortType(type),type));
    if([...el('family').options].some(o=>o.value===selected))el('family').value=selected;
    ceresMap.selected=null;ceresMap.hovered=null;ceresMap.autoFit=true;drawNativeProblem();
  }catch(error){el('summary').textContent=error.message;el('summary').className='error';}
}
for(const id of ['state','family','limit','grouped','segment'])el(id).onchange=()=>{ceresMap.autoFit=true;drawNativeProblem();};
el('window').onchange=loadNativeWindow;
fetch('.solver_inspections/'+captureDataset+'/manifest.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Generate inspection windows first');return r.json();}).then(manifest=>{
  for(const w of manifest.windows)el('window').add(new Option(`${w.index}: frames ${w.frame_start}–${w.frame_end}`,w.file));
  return loadNativeWindow();
}).catch(error=>{el('summary').textContent=error.message;el('summary').className='error';});
const divider=el('divider');
divider.onpointerdown=e=>{divider.setPointerCapture(e.pointerId);divider.onpointermove=event=>{el('recording').style.height=Math.max(100,Math.min(innerHeight-280,innerHeight-event.clientY))+'px';};};
divider.onpointerup=divider.onpointercancel=()=>{divider.onpointermove=null;};
divider.onkeydown=e=>{if(['ArrowUp','ArrowDown'].includes(e.key)){e.preventDefault();el('recording').style.height=Math.max(100,el('recording').clientHeight+(e.key==='ArrowUp'?30:-30))+'px';}};
