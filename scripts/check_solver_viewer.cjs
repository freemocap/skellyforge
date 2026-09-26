const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync('scripts/solver_viewer.html','utf8');
const nodes={};
function element(tag='div'){
 const node={tagName:tag.toUpperCase(),value:'0',checked:true,hidden:false,innerHTML:'',textContent:'',style:{},dataset:{},children:[],on(){},clientWidth:900,clientHeight:400,events:{},
 addEventListener(k,f){(this.events[k]??=[]).push(f)},
 appendChild(child){this.children.push(child);if(child.id)nodes[child.id]=child;},
 replaceChildren(...children){this.children=[];for(const c of children)this.appendChild(c);if(tag==='select'&&children.length)this.value=children[0].value;},
 setAttribute(k,v){(this.attributes??={})[k]=String(v)},
 querySelector(){return this.contentNode??=(element('g'));},
 querySelectorAll(selector){
  if(this.parsedHTML!==this.innerHTML){this.parsedHTML=this.innerHTML;
   this.graphNodes=[...this.innerHTML.matchAll(/data-node="([^"]+)"/g)].map(m=>Object.assign(element('g'),{dataset:{node:m[1]}}));
   this.graphEdges=[...this.innerHTML.matchAll(/data-source="([^"]+)" data-target="([^"]+)"/g)].map(m=>Object.assign(element('path'),{dataset:{source:m[1],target:m[2]}}));
  }
  return selector==='[data-node]'?this.graphNodes:this.graphEdges;
 },
 setPointerCapture(id){this.capture=id},hasPointerCapture(id){return this.capture===id},releasePointerCapture(){this.capture=null},
 getBoundingClientRect(){return {left:0,top:0,width:600,height:160}},classList:{values:new Set(),add(k){this.values.add(k)},remove(k){this.values.delete(k)},contains(k){return this.values.has(k)},toggle(k,force){const enabled=force??!this.values.has(k);if(enabled)this.values.add(k);else this.values.delete(k);return enabled}}};
 return node;
}
for(const match of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"/g))nodes[match[2]]=element(match[1]);
nodes['map-all'].checked=false;
const THREE=require('./vendor/three.min.js');THREE.WebGLRenderer=class{constructor(){this.domElement={addEventListener(){}}}setPixelRatio(){}setSize(){}render(){}};THREE.OrbitControls=class{constructor(){this.target=new THREE.Vector3()}update(){}};
const context={THREE,Plotly:{react(g,t,l){g.traces=t;g._fullLayout=l;return Promise.resolve();},relayout(g,l){g.lastRelayout=l;},Plots:{resize(){}}},document:{createElement:element,addEventListener(){},getElementById:id=>{assert(nodes[id],`Missing DOM id ${id}`);return nodes[id]},documentElement:{style:{setProperty(){}}},body:{classList:{add(){},remove(){}}}},innerWidth:1400,innerHeight:900,devicePixelRatio:1,window:{addEventListener(){}},requestAnimationFrame(){},ResizeObserver:class{constructor(fn){this.fn=fn}observe(){this.fn()}},performance:{now:()=>0},console};
vm.createContext(context);vm.runInContext(html.match(/<script>const EXPERIMENTS=([\s\S]*?)<\/script>/)[0].replace('<script>','').replace('</script>',''),context);
vm.runInContext(['solver_scene.js','solver_layout.js','solver_time_series.js','solver_map.js','solver_problem.js','solver_viewer.js'].map(name=>fs.readFileSync('scripts/'+name,'utf8')).join('\n'),context);
if(vm.runInContext('EXPERIMENTS.some(e=>e.id==="recording_shoulder_linkages")',context)){
 nodes.experiment.value='recording_shoulder_linkages';vm.runInContext('chooseExperiment()',context);
 nodes['body-region'].value='all';vm.runInContext('chooseRegion()',context);
 for(const id of ['lower_sc_relaxed','lower_closer_sc_relaxed']){
  nodes.mode.value=id;vm.runInContext('selectRun();draw()',context);
  assert(nodes['problem-summary'].textContent.includes('2244 parameter blocks'));
  assert(nodes['problem-summary'].textContent.includes('6239 residual blocks'));
  assert(vm.runInContext('ceresMap.nodes.filter(n=>n.id.startsWith("linkage-")).length',context)===6);
  assert(vm.runInContext('ceresMap.nodes.filter(n=>n.id.startsWith("linkage-")).every(n=>n.detail.includes("3 stored values")&&!n.subtitle.includes("undefined"))',context));
  assert(vm.runInContext('ceresMap.nodes.some(n=>n.kind==="chain"&&!selectedMode.problem.relaxed_linkage_children.includes(n.bodyIndex)&&n.connections.some(c=>c.startsWith("linkage-")))',context));
  assert(nodes['displacement-panel'].hidden===false);
  assert(nodes['displacement-series'].traces.length===4);
  for(const index of [0,12,13,14,23]){
   vm.runInContext(`frameIndex=${index};draw()`,context);
   const links=vm.runInContext('current.linkages',context);
   for(const link of links){
    const gap=new THREE.Vector3(...link.child_point).distanceTo(new THREE.Vector3(...link.parent_point));
    assert(Math.abs(gap-Math.hypot(...link.local_displacement))<1e-8);
   }
   assert(vm.runInContext('groups.fitted.children.filter(n=>n.type==="Line"&&n.userData.label.includes("fitted linkage displacement")).length',context)===2);
  }
 }
 console.log('Relaxed shoulders: XYZ blocks, descendant dependencies, native counts, plotted gaps and saved-geometry connectors passed.');
}
for(const exp of vm.runInContext('EXPERIMENTS',context)){
 nodes.experiment.value=exp.id;vm.runInContext('chooseExperiment()',context);
 for(const mode of exp.methods){
  nodes.mode.value=mode.id;vm.runInContext('selectRun()',context);
  for(let body=0;body<exp.bodies.length;body++){
   nodes['plot-body'].value=String(body);vm.runInContext('selectBody();draw()',context);
   assert(nodes['rotation-series'].traces.length===(vm.runInContext('current.bodies[Number(el("plot-body").value)].reference_quaternion!==null',context)?8:4));
   assert(nodes.status.textContent.startsWith(vm.runInContext('current.converged',context)?'CONVERGED':'NOT CONVERGED'));
   vm.runInContext('frameIndex=selectedMode.frames.length-1;draw()',context);
  }
 }
}
nodes.experiment.value='moving';vm.runInContext('chooseExperiment()',context);nodes['parameter-missing'].value='3';vm.runInContext('selectRun()',context);nodes['plot-point'].value='7';vm.runInContext('drawTimeSeries()',context);assert(nodes['point-value'].textContent.includes('Observation missing'));
for(const key of ['x','y','z','w'])nodes['plot-'+key].checked=false;vm.runInContext('drawTimeSeries()',context);assert(nodes['translation-series'].traces.length===0);
console.log('Viewer control smoke passed: all experiments, all methods and segments, first/last frames, missing landmark and component toggles. No browser rendering.');

nodes.experiment.value='linkage';vm.runInContext('chooseExperiment()',context);
for(const count of ['8','4','2','1']){
 nodes['parameter-child_points'].value=count;
 for(const mode of ['independent','connected']){
  nodes.mode.value=mode;vm.runInContext('selectRun()',context);
  nodes['plot-body'].value='1';vm.runInContext('selectBody()',context);
  nodes['plot-point'].value='7';vm.runInContext('draw()',context);
  if(count!=='8')assert(nodes['point-value'].textContent.includes('Observation missing'));
  if(count==='1')assert(nodes.observability.textContent.includes('unconstrained'));
 }
}
console.log('Partial child observations: all four counts and both methods passed.');

nodes.mode.value='connected';nodes['parameter-child_points'].value='2';vm.runInContext('selectRun()',context);
assert(nodes['problem-summary'].textContent.includes('3 parameter blocks (11 stored values / 9 tangent dimensions)'));
assert(nodes['problem-summary'].textContent.includes('10 residual blocks'));
assert(nodes['problem-series'].innerHTML.includes('LandmarkResidual '+String.fromCharCode(215)+' 2'));
assert(nodes['problem-value'].textContent.includes('SAME joint-position'));
nodes.mode.value='independent';vm.runInContext('selectRun()',context);
assert(nodes['problem-summary'].textContent.includes('4 parameter blocks'));
nodes.experiment.value='moving';vm.runInContext('chooseExperiment()',context);
nodes.mode.value='acceleration';vm.runInContext('selectRun()',context);
assert(nodes['problem-series'].innerHTML.includes('QuaternionAccelerationResidual'));
assert(nodes['problem-summary'].textContent.includes('82 parameter blocks'));
console.log('Ceres inspector counts and connection families passed.');

for(const key of ['x','y','z','w'])nodes['plot-'+key].checked=true;
nodes.experiment.value='linked_sequence';vm.runInContext('chooseExperiment()',context);
nodes['parameter-gap'].value='5';nodes.mode.value='temporal';vm.runInContext('selectRun();frameIndex=20;draw()',context);
assert(nodes['problem-summary'].textContent.includes('123 parameter blocks'));
assert(nodes['problem-summary'].textContent.includes('733 residual blocks'));
assert(nodes['problem-series'].innerHTML.includes('Child segment; frames 19, 20, 21'));
assert(nodes.observability.textContent.includes('no landmark residuals'));
nodes['plot-body'].value='1';nodes.mode.value='per_frame';vm.runInContext('selectRun();selectBody();draw()',context);
assert(nodes['translation-value'].textContent.includes('no fitted pose'));
assert(nodes['problem-summary'].textContent.includes('2 parameter blocks'));
assert(nodes['translation-series'].traces[0].y[20]===null);
assert(nodes['translation-series']._fullLayout.shapes.length===2);
console.log('Linked sequence gap: temporal residual connections, missing per-frame pose and plot gaps passed.');

nodes.experiment.value='chain';vm.runInContext('chooseExperiment();frameIndex=20;draw()',context);
assert(nodes['problem-summary'].textContent.includes('164 parameter blocks'));
assert(nodes['problem-summary'].textContent.includes('1100 residual blocks'));
assert(nodes['problem-series'].innerHTML.includes('ChainLandmarkResidual'));
assert((nodes['problem-series'].innerHTML.match(/<path /g)||[]).length===30);
assert(nodes['problem-value'].textContent.includes('No separate joint-position blocks'));
for(const body of ['0','1','2']){nodes['plot-body'].value=body;vm.runInContext('selectBody();draw()',context);}
assert(nodes.status.textContent.includes('Second attachment separation'));
console.log('Three-segment chain: root/upstream residual connections, both linkages and all segment plots passed.');

const mapHost=nodes['problem-series'];
const parameters=vm.runInContext('ceresMap.nodes.filter(n=>n.quaternion)',context);
const selectedId=parameters[0].id;
const pointer={target:{closest:()=>({dataset:{node:selectedId}})},clientX:30,clientY:40,preventDefault(){}};
mapHost.events.pointermove[0](pointer);
assert(nodes['problem-selection'].textContent.includes('HOVER'));
let incident=mapHost.querySelectorAll('[data-source]').filter(e=>e.dataset.source===selectedId||e.dataset.target===selectedId);
assert(incident.length>0&&incident.every(e=>e.style.opacity==='1'));
assert(mapHost.querySelectorAll('[data-source]').filter(e=>!incident.includes(e)).every(e=>e.style.opacity==='0'));
mapHost.events.click[0](pointer);mapHost.events.pointerleave[0]();
assert(nodes['problem-selection'].textContent.includes('PINNED'));
const oldScale=vm.runInContext('ceresMap.scale',context);nodes['map-zoom-in'].onclick();
assert(vm.runInContext('ceresMap.scale',context)>oldScale);
const oldX=vm.runInContext('ceresMap.x',context);
const background={target:{closest:()=>null},button:0,pointerId:1,clientX:10,clientY:10,preventDefault(){}};
mapHost.events.pointerdown[0](background);mapHost.events.pointermove[0]({...background,clientX:70,clientY:30});mapHost.events.pointerup[0](background);
assert(vm.runInContext('ceresMap.x',context)===oldX+60);
nodes['map-clear'].onclick();nodes['map-fit'].onclick();
assert(vm.runInContext('ceresMap.selected',context)===null);
assert(vm.runInContext('ceresMap.autoFit',context)===true);
console.log('Interactive Ceres map: focused edges, pinning, zoom, pan, clearing and fit passed.');

nodes.experiment.value='displacement';vm.runInContext('chooseExperiment();frameIndex=20;draw()',context);
assert(nodes['problem-summary'].textContent.includes('205 parameter blocks'));
assert(nodes['problem-summary'].textContent.includes('1220 residual blocks'));
assert(nodes['problem-series'].innerHTML.includes('DisplacementPriorResidual'));
assert(nodes['problem-series'].innerHTML.includes('DisplacementAccelerationResidual'));
assert(vm.runInContext('ceresMap.nodes.filter(n=>n.kind==="displacement").length',context)===3);
assert(vm.runInContext('ceresMap.edges.length',context)===48);
assert(nodes['displacement-panel'].hidden===false);
assert(nodes['displacement-series'].traces.length===2);
assert(nodes['displacement-series'].traces[1].y[20]===20);
nodes.mode.value='fixed';vm.runInContext('selectRun()',context);
assert(nodes['problem-summary'].textContent.includes('164 parameter blocks'));
assert(nodes['displacement-series'].traces[0].y.every(value=>value===0));
console.log('Bounded displacement: scalar blocks/residuals, graph connections, trace and fixed baseline passed.');

nodes.experiment.value='displacement_gap';vm.runInContext('chooseExperiment();frameIndex=20;draw()',context);
assert(nodes['problem-summary'].textContent.includes('205 parameter blocks'));
assert(nodes['problem-summary'].textContent.includes('1180 residual blocks'));
assert(vm.runInContext('ceresMap.edges.length',context)===39);
assert(nodes.observability.textContent.includes('not middle roll'));
assert(nodes['displacement-series']._fullLayout.shapes.length===2);
assert(vm.runInContext('current.bodies[1].observed.every(p=>p===null)',context));
nodes.mode.value='fixed';vm.runInContext('selectRun()',context);
assert(nodes['problem-summary'].textContent.includes('1100 residual blocks'));
console.log('Combined displacement/gap: omitted observations, scalar connections and shaded plots passed.');

nodes['map-types-parameters'].onclick();
assert(vm.runInContext('ceresMap.nodes.every(n=>!n.connections)',context));
assert(vm.runInContext('ceresMap.edges.length',context)===0);
nodes['map-types-motion'].onclick();
assert(vm.runInContext('ceresMap.nodes.every(n=>["angular","linear","displacement_motion"].includes(n.kind))',context));
nodes['map-types-all'].onclick();
assert(vm.runInContext('ceresMap.edges.length',context)>0);
nodes['map-type-quaternion'].checked=false;nodes['map-type-quaternion'].events.change[0]();
assert(vm.runInContext('ceresMap.nodes.every(n=>n.kind!=="quaternion")',context));
assert(vm.runInContext('ceresMap.edges.every(e=>ceresMap.nodes.some(n=>n.id===e.source)&&ceresMap.nodes.some(n=>n.id===e.target))',context));
nodes['map-types-all'].onclick();
for(const id of ['problem','displacement','translation','rotation','point']){
 nodes[id+'-collapse'].onclick();assert(nodes[id+'-panel'].classList.contains('collapsed-panel'));
 nodes[id+'-collapse'].onclick();assert(!nodes[id+'-panel'].classList.contains('collapsed-panel'));
}
assert(vm.runInContext('groups.fitted.children.some(n=>n.geometry?.type==="CylinderGeometry")',context));
assert(vm.runInContext('groups.fitted.children.some(n=>n.geometry?.type==="SphereGeometry"&&n.userData.label?.includes("attachment, not landmark"))',context));
console.log('Map type filters, panel collapse controls and distinct segment/attachment geometry passed.');

nodes.experiment.value='branching';vm.runInContext('chooseExperiment();frameIndex=10;draw()',context);
assert(nodes['problem-summary'].textContent.includes('164 parameter blocks'));
assert(vm.runInContext('ceresMap.nodes.filter(n=>n.connections&&n.bodyLabel==="Branch B"&&n.index!==null).every(n=>n.connections.includes(`q-${n.index}-0`)&&n.connections.includes(`q-${n.index}-2`)&&!n.connections.includes(`q-${n.index}-1`))',context));
vm.runInContext('frameIndex=20;draw()',context);
assert(nodes.observability.textContent.includes('Branch A'));
assert(vm.runInContext('current.diagnostics["Second attachment separation (mm)"]<1e-9',context));
console.log('Branching map: shared parent, no sibling quaternion dependency, missing branch observations passed.');

nodes.experiment.value='tree';vm.runInContext('chooseExperiment();frameIndex=10;draw()',context);
assert(nodes['problem-summary'].textContent.includes('126 parameter blocks'));
assert(nodes['problem-summary'].textContent.includes('954 residual blocks'));
assert(vm.runInContext('ceresMap.nodes.filter(n=>n.connections&&n.bodyLabel==="Right branch"&&n.index!==null).every(n=>n.connections.includes(`q-${n.index}-2`)&&n.connections.includes(`q-${n.index}-4`)&&!n.connections.includes(`q-${n.index}-3`))',context));
console.log('Five-segment tree: ancestor-only map connections and block counts passed.');

nodes.experiment.value='torso';vm.runInContext('chooseExperiment();frameIndex=10;draw()',context);
assert(nodes['problem-summary'].textContent.includes('282 residual blocks'));
assert(vm.runInContext('current.bodies.reduce((n,b)=>n+b.observed.filter(p=>p!==null).length,0)',context)===4);
assert(vm.runInContext('ceresMap.nodes.filter(n=>n.kind==="pose_prior").length',context)===12);
assert(nodes['problem-series'].innerHTML.includes('RelativePoseResidual'));
nodes.mode.value='no_prior';vm.runInContext('selectRun()',context);
assert(nodes['problem-summary'].textContent.includes('198 residual blocks'));
assert(vm.runInContext('ceresMap.nodes.every(n=>n.kind!=="pose_prior")',context));
console.log('Sparse torso: four observations, relative pose blocks on/off and native counts passed.');

if(vm.runInContext('EXPERIMENTS.some(e=>e.id==="recording_torso")',context)){
 nodes.experiment.value='recording_torso';vm.runInContext('chooseExperiment()',context);nodes.mode.value='rest_prior';vm.runInContext('selectRun();frameIndex=200;draw()',context);
 assert(nodes.status.textContent.includes('not available (real recording)'));
 assert(nodes['rotation-value'].textContent.includes('no known reference'));
 assert(nodes['rotation-series'].traces.every(t=>!t.name.includes('known')));
 assert(vm.runInContext('groups.truth.children.length',context)===0);
 assert(vm.runInContext('current.bodies.reduce((n,b)=>n+b.observed.filter(p=>p!==null).length,0)',context)===4);
 assert(nodes['problem-summary'].textContent.includes('3012 residual blocks'));
 assert(vm.runInContext('current.diagnostics["Recording frame"]',context)===200);
 console.log('Real recording: no invented reference poses, four targets and saved frame numbering passed.');
}


if(vm.runInContext('experiment.annotated_views?.length',context)){
 const pending=number=>vm.runInContext(`Array.from(annotatedFrames.values()).find(e=>e.image.src.includes('/camera0/${String(number).padStart(6,'0')}.jpg'))`,context);
 assert(nodes['video-panel'].hidden===false);
 pending(200).image.onload();
 assert(nodes['annotated-frame'].src.includes('/000200.jpg'));
 assert(vm.runInContext('groups.contextSegments.children.length>0&&groups.contextKeypoints.children.length>0',context));
 vm.runInContext('frameIndex=201;draw()',context);
 assert(nodes['annotated-frame'].hidden===false);
 assert(nodes['annotated-frame'].src.includes('/000200.jpg'));
 const obsolete=pending(201);
 vm.runInContext('frameIndex=202;draw()',context);
 obsolete.image.onload();assert(nodes['annotated-frame'].src.includes('/000200.jpg'));
 pending(202).image.onload();assert(nodes['annotated-frame'].src.includes('/000202.jpg'));
 assert(nodes['video-value'].textContent.includes('Recording frame 202'));
 nodes['video-camera'].value='2';nodes['video-camera'].events.change[0]();
 assert(nodes['annotated-frame'].src.includes('/camera0/000202.jpg'));
 vm.runInContext('Array.from(annotatedFrames.values()).find(e=>e.image.src.includes("/camera2/000202.jpg")).image.onload()',context);
 assert(nodes['annotated-frame'].src.includes('/camera2/000202.jpg'));
 nodes['video-collapse'].onclick();assert(nodes['video-panel'].classList.contains('collapsed-panel'));
 nodes['video-collapse'].onclick();
 assert(vm.runInContext('groups.fitted.children.some(n=>n.geometry?.type==="CylinderGeometry"&&n.geometry.parameters.radiusTop<n.geometry.parameters.radiusBottom)',context));
 console.log('Decoded frame swaps retain visible images, reject late loads, switch cameras; fitted bones are tapered.');
}

if(vm.runInContext('EXPERIMENTS.find(e=>e.id==="recording_torso")?.methods.some(m=>m.id==="flexible")',context)){
 nodes.experiment.value='recording_torso';vm.runInContext('chooseExperiment()',context);
 nodes.mode.value='flexible';vm.runInContext('selectRun();frameIndex=192;draw()',context);
 assert(nodes['problem-summary'].textContent.includes('1728 parameter blocks'));
 assert(nodes['problem-summary'].textContent.includes('3872 residual blocks'));
 assert(nodes['length-panel'].hidden===false);
 assert(nodes['length-series'].traces.length===4);
 assert(vm.runInContext('ceresMap.nodes.filter(n=>n.kind==="length").length',context)===6);
 assert(vm.runInContext('ceresMap.nodes.filter(n=>n.kind==="chain"&&n.bodyLabel==="left_clavicle").every(n=>n.connections.includes(`length-${n.index}-1`)&&n.connections.includes(`length-${n.index}-2`))',context));
 nodes['map-types-parameters'].onclick();assert(vm.runInContext('ceresMap.nodes.some(n=>n.kind==="length")',context));nodes['map-types-all'].onclick();
 console.log('Flexible spine: scalar blocks, ancestor connections, native counts and length traces passed.');
}

if(vm.runInContext('EXPERIMENTS.some(e=>e.id==="recording_body")',context)){
 nodes.experiment.value='recording_body';vm.runInContext('chooseExperiment()',context);
 assert(vm.runInContext('experiment.bodies.length',context)===61);
 assert(vm.runInContext('current.bodies.filter(b=>b.mechanical_model==="axial").length',context)===2);
 assert(vm.runInContext('(selectedMode.summary["Maximum attachment equation error (mm)"]??selectedMode.summary["Maximum attachment error (mm)"])<1e-8',context));
 assert(vm.runInContext('selectedMode.summary["Ceres parameter blocks"]',context)===vm.runInContext('(experiment.bodies.length+3)*selectedMode.frames.length',context));
 assert(nodes['problem-summary'].textContent.includes(String(vm.runInContext('selectedMode.summary["Ceres residual blocks"]',context))+' residual blocks'));
 nodes['body-region'].value='Head and neck';nodes['body-region'].events.change[0]();
 const skull=vm.runInContext('experiment.bodies.findIndex(b=>b.id==="skull")',context);
 nodes['body-focus'].value=String(skull);nodes['body-focus'].events.change[0]();
 assert(vm.runInContext('ceresMap.nodes.filter(n=>n.connections).every(n=>experiment.bodies[n.bodyIndex].id==="skull")',context));
 assert(vm.runInContext('ceresMap.nodes.some(n=>n.kind==="length")',context));
 assert(vm.runInContext('ceresMap.edges.every(e=>ceresMap.nodes.some(n=>n.id===e.source))',context));
 assert(vm.runInContext('axisLabels.length',context)===1);
 assert(nodes['axis-labels'].children[0].textContent==='skull');
 assert(vm.runInContext('groups.axes.children.some(n=>n.geometry?.type==="SphereGeometry"&&n.userData.label.includes("skull"))',context));
 nodes['segment-colors'].value='support';nodes['segment-colors'].events.change[0]();
 assert(nodes['segment-legend'].textContent.includes('not a global observability test'));
 assert(vm.runInContext(`segmentStyle(current.bodies[${skull}],${skull}).color`,context)===0x63d6a3);
 assert(vm.runInContext('groups.truth.children.length',context)===0);
 nodes['recording-frame'].value='192';nodes['recording-frame'].events.change[0]();
 assert(vm.runInContext('current.diagnostics["Recording frame"]',context)===192);
 assert(nodes.clock.textContent.includes('Recording frame 192'));
 for(const choice of vm.runInContext('experiment.methods',context)){
  nodes.mode.value=choice.id;vm.runInContext('selectRun()',context);
  nodes['body-region'].value='Trunk';nodes['body-region'].events.change[0]();
  if(vm.runInContext('selectedMode.settings.free_axial_lengths===true',context)){
   assert(vm.runInContext('!ceresMap.nodes.some(n=>n.kind==="length_prior"||n.kind==="length_motion")',context));
   assert(vm.runInContext('ceresMap.nodes.filter(n=>n.kind==="length").every(n=>n.subtitle.includes("Infinity"))',context));
   assert(nodes['problem-summary'].textContent.includes(String(vm.runInContext('selectedMode.summary["Ceres residual blocks"]',context))+' residual blocks'));
   continue;
  }
  const scales=vm.runInContext('({short:selectedMode.settings.length_prior_fraction,long:selectedMode.settings.lengthening_prior_fraction??selectedMode.settings.length_prior_fraction,ref:selectedMode.settings.axial_reference_lengths[selectedMode.problem.axial_segments[0]]})',context);
  assert(vm.runInContext('ceresMap.nodes.filter(n=>n.kind==="length_prior").length>0',context));
  const detail=vm.runInContext('ceresMap.nodes.find(n=>n.kind==="length_prior").detail',context);
  assert(detail.includes('shortening scale '+scales.short*scales.ref));
  assert(detail.includes('lengthening scale '+scales.long*scales.ref));
 }
 console.log('Full body: native counts, exact attachments, region/segment filtering with ancestors, labeled axis origins and support colors passed.');
}

if(vm.runInContext('EXPERIMENTS.find(e=>e.id==="recording_body")?.methods.some(m=>m.id==="chest_line")',context)){
 nodes.experiment.value='recording_body';vm.runInContext('chooseExperiment()',context);
 for(const method of ['free_lengths','chest_line']){
  nodes.mode.value=method;vm.runInContext('selectRun();draw()',context);
  assert(nodes['chest-line-panel'].hidden===false);
  assert(nodes['chest-line-series'].traces.length===4);
  assert(vm.runInContext('groups.chestCenterline.children.some(n=>n.userData.label.includes("Fitted chest_center"))',context));
  assert(vm.runInContext('groups.chestCenterline.children.every(n=>n.userData.label)',context));
  nodes['body-region'].value='Trunk';nodes['body-region'].events.change[0]();
  assert(vm.runInContext('ceresMap.nodes.filter(n=>n.kind==="chest_line").length',context)===(method==='chest_line'?3:0));
  assert(nodes['problem-summary'].textContent.includes(String(vm.runInContext('selectedMode.summary["Ceres residual blocks"]',context))+' residual blocks'));
 }
 nodes.chestCenterline.checked=false;nodes.chestCenterline.events.change[0]();
 assert(vm.runInContext('groups.chestCenterline.visible',context)===false);
 nodes.chestCenterline.checked=true;nodes.chestCenterline.events.change[0]();
 nodes['chest-line-collapse'].onclick();assert(nodes['chest-line-panel'].classList.contains('collapsed-panel'));
 nodes['chest-line-collapse'].onclick();nodes['chest-line-expand'].onclick();
 assert(nodes['chest-line-panel'].classList.contains('expanded-plot'));
 console.log('Chest centerline: overlay identities, display toggle, displacement plots, panel controls and actual Ceres block counts passed.');
}

if(vm.runInContext('EXPERIMENTS.some(e=>e.id==="recording_shoulders")',context)){
 nodes.experiment.value='recording_shoulders';vm.runInContext('chooseExperiment()',context);
 const thorax=vm.runInContext('experiment.bodies.findIndex(b=>b.id==="thoracic")',context);
 for(const method of ['current_sc','lower_sc','lower_closer_sc']){
  nodes.mode.value=method;vm.runInContext('selectRun()',context);
  for(const index of [0,12,13,14,23]){
   vm.runInContext(`frameIndex=${index};draw()`,context);
   for(const side of ['left','right']){
    const name=side+'_sternoclavicular';
    const marker=vm.runInContext(`groups.fitted.children.find(n=>n.userData.label==='thoracic / fitted ${name} (attachment, not landmark)')`,context);
    const expected=vm.runInContext(`current.bodies[${thorax}].fitted[experiment.bodies[${thorax}].landmark_names.indexOf('${name}')]`,context);
    assert(marker.position.distanceTo(new THREE.Vector3(...expected))<1e-8);
    const child=vm.runInContext(`experiment.bodies.findIndex(b=>b.id==='${side}_clavicle')`,context);
    const origin=vm.runInContext(`current.bodies[${child}].translation`,context);
    assert(marker.position.distanceTo(new THREE.Vector3(...origin))<1e-8);
   }
  }
  assert(nodes['problem-summary'].textContent.includes(String(vm.runInContext('selectedMode.summary["Ceres residual blocks"]',context))+' residual blocks'));
 }
 console.log('Shoulder variants: per-method reference attachments agree with fitted landmarks and exact clavicle origins at upright/bending frames.');
}

if(vm.runInContext('EXPERIMENTS.some(e=>e.id==="recording_spine_equality")',context)){
 nodes.experiment.value='recording_spine_equality';vm.runInContext('chooseExperiment()',context);
 nodes.mode.value='equal_spine_lengths';vm.runInContext('selectRun();draw()',context);
 assert(nodes['problem-summary'].textContent.includes('2244 parameter blocks'));
 assert(nodes['problem-summary'].textContent.includes('6273 residual blocks'));
 assert(vm.runInContext('ceresMap.nodes.filter(n=>n.kind==="length_equality").length',context)===3);
 assert(vm.runInContext('ceresMap.nodes.filter(n=>n.kind==="length_equality").every(n=>n.connections.length===2&&n.connections.every(id=>id.startsWith("length-")))',context));
 const thoracic=vm.runInContext('experiment.bodies.findIndex(b=>b.id==="thoracic")',context);
 nodes['body-focus'].value=String(thoracic);vm.runInContext('draw()',context);
 assert(vm.runInContext('ceresMap.nodes.filter(n=>n.kind==="length_equality").length',context)===3);
 console.log('Length equality: two scalar dependencies, no extra parameters, native residual counts, and either-segment filtering passed.');
}
