/* Read-only real-data UI check. Real Three geometry; mocked browser rendering. */
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync('scripts/recording_comparison.html','utf8');
const data=JSON.parse(html.match(/const RECORDING_COMPARISON=(.*?);<\/script>/s)[1]);
const nodes={},images=[];
function element(tag='div'){
 const n={tagName:tag.toUpperCase(),children:[],style:{setProperty(k,v){this[k]=v}},dataset:{},checked:false,value:'',hidden:false,clientWidth:900,clientHeight:650,events:{},textContent:'',
 appendChild(c){this.children.push(c);return c},replaceChildren(...c){this.children=c},setAttribute(k,v){this[k]=v},addEventListener(k,f){(this.events[k]??=[]).push(f)},getBoundingClientRect(){return {left:0,top:0,bottom:650,width:900,height:650}},setPointerCapture(i){this.capture=i},hasPointerCapture(i){return this.capture===i},releasePointerCapture(){this.capture=null},classList:{toggle(){},add(){},remove(){}}};return n;
}
for(const match of html.matchAll(/<(\w+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)){nodes[match[3]]=element(match[1]);nodes[match[3]].checked=/\bchecked\b/.test(match[2]);}
nodes.region.value='all';nodes.speed.value='1';
const THREE=require('./vendor/three.min.js');THREE.WebGLRenderer=class{constructor(){this.domElement=element('canvas')}setPixelRatio(){}setSize(){}render(){}};THREE.OrbitControls=class{constructor(){this.target=new THREE.Vector3()}update(){}};
const context={RECORDING_COMPARISON:data,THREE,console,devicePixelRatio:1,innerWidth:1400,innerHeight:900,performance:{now:()=>0},requestAnimationFrame(){},ResizeObserver:class{constructor(fn){this.fn=fn}observe(){this.fn()}},Image:class{constructor(){images.push(this)}decode(){return Promise.resolve()}},document:{createElement:element,getElementById(id){assert(nodes[id],id);return nodes[id]},querySelector(){return element('aside')},addEventListener(){},documentElement:element('html'),body:element('body')},window:{addEventListener(){}}};
vm.createContext(context);vm.runInContext(['recording_comparison_scene.js','recording_comparison_table.js','recording_comparison.js'].map(p=>fs.readFileSync('scripts/'+p,'utf8')).join('\n'),context);
const run=source=>vm.runInContext(source,context);
assert(run('reviewState.solutions.filter(s=>s.enabled).length')===2);
if(data.solutions.some(s=>s.id==='equal_spine_lengths')){
 assert(run('describeComparisonFit(reviewData.solutions.find(s=>s.id==="equal_spine_lengths")).spine')==='Free + equal-length prior');
 assert(nodes['inspector-lengths'].textContent.includes('sacrolumbar'));
 assert(run('reviewData.solutions.filter((s,i)=>reviewState.solutions[i].enabled).every(s=>["lower_sc_relaxed","equal_spine_lengths"].includes(s.id))'));
}
// Every saved landmark is placed verbatim, across all fits and every frame.
for(let i=0;i<data.times.length;i++){
 run(`seekReview(${i})`);
 const views=run('viewer.solutions');
 for(let s=0;s<data.solutions.length;s++)for(let b=0;b<views[s].bodies.length;b++){
  const pose=data.solutions[s].frames[i].bodies[b];
  views[s].bodies[b].points.forEach((p,j)=>assert(p.position.distanceTo(new THREE.Vector3(...pose.fitted[j]))<1e-9));
  const definition=data.solutions[s].definitions[b];
  views[s].bodies[b].bones.forEach((mesh,j)=>{
   const local=[...definition.attachments[j].position];local[2]*=pose.axial_scale;
   const q=pose.quaternion,end=new THREE.Vector3(...local).applyQuaternion(new THREE.Quaternion(q[1],q[2],q[3],q[0])).add(new THREE.Vector3(...pose.translation));
   const start=new THREE.Vector3(...pose.translation);const length=start.distanceTo(end);
   if(length<1e-8){assert(!mesh.visible);return;}
   assert(Math.abs(mesh.scale.y-length)<1e-8);
   const distal=new THREE.Vector3(0,mesh.scale.y/2,0).applyQuaternion(mesh.quaternion).add(mesh.position);
   assert(distal.distanceTo(end)<1e-8);
  });
 }
}
nodes['hide-all'].onclick();assert(run('viewer.solutions.every(s=>!s.group.visible)'));
nodes['reset-solutions'].onclick();assert(run('viewer.solutions.filter(s=>s.group.visible).length')===2);
run('fitTable.rows[0].solo.onclick()');assert(run('reviewState.solutions[0].enabled&&reviewState.solutions.filter(s=>s.enabled).length===1'));
nodes['reset-solutions'].onclick();
assert(run('fitTable.rows.length')===data.solutions.length);
const before=run('JSON.stringify(reviewState.solutions)');run('fitTable.rows[1].inspect.onclick()');
assert(run('JSON.stringify(reviewState.solutions)')===before);assert(nodes['fit-inspector'].open);
assert(nodes['inspector-title'].textContent===data.solutions[1].label);
assert(nodes['inspector-settings'].textContent.includes('direct_mapping_sources'));
run('fitTable.rows[0].opacity.value=35;fitTable.rows[0].opacity.onchange()');assert(run('reviewState.solutions[0].opacity')===.35);
assert(run('describeComparisonFit(reviewData.solutions.find(s=>s.id==="lower_sc_relaxed")).shoulders')==='Relaxed');
assert(run('describeComparisonFit(reviewData.solutions.find(s=>s.id==="lower_sc")).shoulders')==='Exact');
nodes['region'].value='Left arm';nodes['region'].onchange();assert(run('viewer.solutions[0].bodies.every((b,i)=>b.g.visible===(reviewData.solutions[0].definitions[i].region==="Left arm"))'));
nodes['region'].value='all';nodes['region'].onchange();
nodes['timeline'].value=12;nodes['timeline'].oninput();assert(run('reviewIndex')===12);assert(nodes['frame-label'].textContent.includes(String(data.frame_ids[12])));
run('seekReview(0);toggleReview();advanceReview(500)');assert(run('reviewIndex')===3);
nodes.loop.checked=false;run('advanceReview(100000)');assert(!run('reviewPlaying'));assert(run('reviewIndex')===data.times.length-1);
nodes.loop.checked=true;run('seekReview(0);toggleReview();advanceReview(100000)');assert(run('reviewPlaying'));assert(run('reviewIndex')>=0&&run('reviewIndex')<data.times.length);
run('pauseReview();seekReview(0)');
const initial=run('desiredVideo');run('seekReview(1)');const latest=run('desiredVideo');
async function verifyImages(){
 const old=images.find(i=>i.src===initial),next=images.find(i=>i.src===latest);
 next.onload();await new Promise(resolve=>setImmediate(resolve));assert(nodes['video-image'].dataset.frame===String(data.frame_ids[1]));
 old.onload();await new Promise(resolve=>setImmediate(resolve));assert(nodes['video-image'].dataset.frame===String(data.frame_ids[1]));
 assert(nodes['video-status'].textContent.includes('annotated'));
 console.log(`Recording comparison passed: ${data.solutions.length} fits × ${data.times.length} frames; exact geometry, overlays, isolate/reset, body filtering, timeline, looping, and stale-image rejection.`);
}
verifyImages().catch(error=>{console.error(error);process.exitCode=1});
