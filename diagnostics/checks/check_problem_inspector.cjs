const fs=require('fs'),vm=require('vm'),assert=require('assert');
const nodes={};
function element(){return {value:'',checked:false,style:{},children:[],options:[],innerHTML:'',textContent:'',clientWidth:1200,clientHeight:700,dataset:{},
 addEventListener(){},setAttribute(){},appendChild(c){this.children.push(c);if(c.id)nodes[c.id]=c;},add(o){this.options.push(o);if(this.options.length===1)this.value=o.value;},replaceChildren(...c){this.options=c;this.value=c[0]?.value??'';},
 querySelector(){return {setAttribute(){}};},
 querySelectorAll(selector){if(this.last!==this.innerHTML){this.last=this.innerHTML;this.graphNodes=[...this.innerHTML.matchAll(/data-node="([^"]+)"/g)].map(m=>({...element(),dataset:{node:m[1]}}));this.graphEdges=[...this.innerHTML.matchAll(/data-source="([^"]+)" data-target="([^"]+)"/g)].map(m=>({...element(),dataset:{source:m[1],target:m[2]}}));}return selector==='[data-node]'?this.graphNodes:this.graphEdges;},
 setPointerCapture(){},hasPointerCapture(){return false;},releasePointerCapture(){},getBoundingClientRect(){return {left:0,top:0};}};}
const el=id=>nodes[id]??=(element());
el('state').value='final';el('limit').value='30';
const context={el,document:{createElement:element},Option:class{constructor(text,value){this.text=text;this.value=value;}},
 ResizeObserver:class{observe(){}},innerHeight:1000,console,
 fetch:async path=>({ok:true,json:async()=>JSON.parse(fs.readFileSync('.test-artifacts/viewers/'+path,'utf8'))})};
vm.createContext(context);
vm.runInContext(fs.readFileSync('skellyforge/tools/viewer/web/solver_map.js','utf8'),context);
vm.runInContext(fs.readFileSync('skellyforge/tools/viewer/web/problem_inspector.js','utf8'),context);
const run=code=>vm.runInContext(code,context);
(async()=>{
 await new Promise(resolve=>setImmediate(resolve));
 assert(!el('summary').className);
 const manifest=JSON.parse(fs.readFileSync('.test-artifacts/viewers/.solver_inspections/test/manifest.json'));
 for(const w of manifest.windows){
  el('window').value=w.file;await run('loadNativeWindow()');
  const data=JSON.parse(fs.readFileSync('.test-artifacts/viewers/.solver_inspections/test/'+w.file));
  for(const state of ['initial','final']){
   el('segment').value='';el('state').value=state;el('family').value='';el('limit').value='0';run('drawNativeProblem()');
   const actual=run('ceresMap.edges.map(e=>[e.source,e.target])');
   const expected=data[state].residuals.flatMap(r=>r.parameter_ids.map(p=>['p'+p,'r'+r.id]));
   assert(JSON.stringify(JSON.parse(JSON.stringify(actual)).sort())===JSON.stringify(expected.sort()),'Rendered connections differ from native block connections');
   assert.strictEqual(run('ceresMap.nodes.length'),data[state].parameters.length+data[state].residuals.length);
   const before=run('JSON.stringify([ceresMap.width,ceresMap.height,ceresMap.nodes.map(n=>[n.x,n.y])])');
   run('ceresMap.hovered=ceresMap.nodes[0].id;highlightCeresMap()');
   assert.strictEqual(run('JSON.stringify([ceresMap.width,ceresMap.height,ceresMap.nodes.map(n=>[n.x,n.y])])'),before);
   assert(el('problem-selection').textContent.includes('ambient_size'));
  }
  el('family').value=data.final.residuals[0].purpose;el('limit').value='10';run('drawNativeProblem()');
  assert(run('ceresMap.nodes.filter(n=>n.connections).length')<=10);
  run('zoomCeresMap(1.3);fitCeresMap()');
  el('family').value='';el('limit').value='0';el('grouped').checked=true;run('drawNativeProblem()');
  assert(run('ceresMap.nodes.every(n=>!n.title.startsWith("Parameter ")&&!n.title.startsWith("Residual "))'));
  assert(run('ceresMap.nodes.filter(n=>n.connections).reduce((sum,n)=>sum+JSON.parse(n.detail).length,0)')===data.final.residuals.length);
  assert(el('problem-series').innerHTML.includes('Active solve'));
  if(w.frame_start<w.active_start)assert(el('problem-series').innerHTML.includes('Fixed history'));
  el('segment').value=String(data.segment_names.indexOf('thoracic'));run('drawNativeProblem()');
  assert(run('ceresMap.nodes.filter(n=>!n.connections).every(n=>n.title==="thoracic")'));
  assert(run('ceresMap.edges.every(e=>ceresMap.nodes.some(n=>n.id===e.source))'));
  el('grouped').checked=false;el('segment').value='';
 }
 console.log('Native inspector passed: three windows, both states, every native edge, counts, stable hover, type/cost filtering, zoom and fit.');
})().catch(e=>{console.error(e);process.exitCode=1;});
