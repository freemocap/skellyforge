/* Real Three geometry with a mocked DOM; no alternate fit mathematics. */
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const base=process.argv[2]||'http://127.0.0.1:8775';
const nodes={},images=[];
function element(){return {style:{setProperty(){}},hidden:false,checked:false,value:'',textContent:'',clientWidth:1000,clientHeight:700,userData:{},appendChild(){},setAttribute(){},replaceChildren(...xs){this.options=xs;this.value=xs[0]?.value??'';},addEventListener(){},getBoundingClientRect(){return {left:0,top:0,width:1000,height:700};},setPointerCapture(){},classList:{add(){},remove(){}}};}
const el=id=>nodes[id]??=element();
for(const id of ['fit','video','loop'])el(id).checked=true;el('speed').value='1';
const THREE=require('./vendor/three.min.js');
THREE.WebGLRenderer=class{constructor(){this.domElement=element();}setPixelRatio(){}setSize(){}render(){}};
THREE.OrbitControls=class{constructor(){this.target=new THREE.Vector3();}update(){}};
const context={THREE,document:{getElementById:el,documentElement:element(),body:element(),addEventListener(){}},devicePixelRatio:1,innerWidth:1400,innerHeight:900,
 ResizeObserver:class{observe(){}},Option:class{constructor(label,value){this.label=label;this.value=value;}},
 performance:{now:()=>0},requestAnimationFrame(){},console,
 Image:class{constructor(){images.push(this);}decode(){return Promise.resolve();}},
 fetch:path=>fetch(base+path)};
vm.createContext(context);vm.runInContext(fs.readFileSync('skellyforge/tools/viewer/simple/simple.js','utf8'),context);
const run=code=>vm.runInContext(code,context);
function close(a,b){assert(a.distanceTo(b)<1e-7,`${a.toArray()} != ${b.toArray()}`);}
(async()=>{
 // Let startup's asynchronous load finish before explicit dataset switching.
 await new Promise(resolve=>setTimeout(resolve,1000));
 for(const dataset of ['synthetic','test','sample']){
  await run(`load('${dataset}')`);
  assert.equal(el('loading').hidden,true,el('loading').textContent);
  const count=run('times.length');assert.equal(count,{synthetic:60,test:222,sample:1108}[dataset]);
  for(const frame of [0,Math.floor(count/2),count-1]){
   run(`seek(${frame})`);
   for(const item of run('sticks')){
    if(!item.mesh.visible)continue;
    const along=new THREE.Vector3(0,item.mesh.scale.y/2,0).applyQuaternion(item.mesh.quaternion);
    close(item.mesh.position.clone().sub(along),new THREE.Vector3(...item.a(frame)));
    close(item.mesh.position.clone().add(along),new THREE.Vector3(...item.b(frame)));
   }
   for(const axis of run('axes')){
    if(!axis.group.visible)continue;
    close(axis.group.position,new THREE.Vector3(...axis.origin(frame)));
    axis.arrows.forEach((arrow,j)=>close(new THREE.Vector3(0,1,0).applyQuaternion(arrow.quaternion),new THREE.Vector3(...axis.basis(frame)[j]).normalize()));
    assert(axis.marker.userData.label.includes('origin'));
   }
   assert(el('frame').textContent.includes('Frame '+frame));
  }
  el('fit').checked=false;run('draw()');assert.equal(run('groups.fit.visible'),false);el('fit').checked=true;
  for(const layer of ['fitAxes','referenceAxes']){
   el(layer).checked=true;run('draw()');assert(run(`groups.${layer}.visible`));
   el(layer).checked=false;run('draw()');assert(!run(`groups.${layer}.visible`));
  }
  assert.equal(el('video-panel').hidden,dataset==='synthetic');
 }
 // An older decoded video frame must not replace a newer requested frame.
 run('seek(1)');const old=images.at(-1);run('seek(2)');const latest=images.at(-1);
 latest.onload();await new Promise(resolve=>setImmediate(resolve));const shown=el('video-image').src;
 old.onload();await new Promise(resolve=>setImmediate(resolve));assert.equal(el('video-image').src,shown);assert(shown.includes('000002.jpg'));
 // Last-frame playback refreshes geometry and stops when looping is disabled.
 el('loop').checked=false;run('seek(times.length-2)');el('play').onclick();run('animate(10000000)');assert.equal(run('index'),1107);assert.equal(run('playing'),false);assert(el('frame').textContent.includes('1107'));
 console.log('Simple viewer passed: all datasets, exact rod endpoints, saved axis bases/origins, independent axis toggles, frame seeking, video race handling and playback completion.');
})().catch(error=>{console.error(error);process.exitCode=1;});
