const el=id=>document.getElementById(id);
const scene=new THREE.Scene();scene.background=new THREE.Color(0x1a1a2e);
const camera=new THREE.PerspectiveCamera(50,1,1,100000);camera.up.set(0,0,1);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));el('scene').appendChild(renderer.domElement);
const controls=new THREE.OrbitControls(camera,renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff,.7));const light=new THREE.DirectionalLight(0xffffff,.5);light.position.set(1,1,1);scene.add(light);
const v=p=>new THREE.Vector3(...p);
const names=Object.keys(DATA.lengths),layers={};
for(const [name,color] of Object.entries({independent:0xffa860,connected:0x888888,fitted:0x00e5ff})) {
  const group=new THREE.Group(), meshes={};scene.add(group);
  for(const n of names) {const mesh=SkeletonGeometry.cylinder(DATA.lengths[n],name==='independent'?4:2.5,color,1);group.add(mesh);meshes[n]=mesh;}
  layers[name]={group,meshes};
}
const pointNames=[...new Set(DATA.frames.flatMap(f=>Object.keys(f.points)))];
const pointGroup=new THREE.Group();scene.add(pointGroup);const pointMeshes={};
const sphere=new THREE.SphereGeometry(4,8,8),material=new THREE.MeshLambertMaterial({color:0xffdd66});
for(const name of pointNames) {const mesh=new THREE.Mesh(sphere,material);pointGroup.add(mesh);pointMeshes[name]=mesh;}
const axes=[0xff5555,0x55ff77,0x5588ff].map(color=>{const a=new THREE.ArrowHelper(new THREE.Vector3(1,0,0),new THREE.Vector3(),80,color,12,6);scene.add(a);return a;});
for(const name of names) {const o=document.createElement('option');o.value=o.textContent=name;el('segment').appendChild(o);}
el('segment').value='thoracic';el('source').textContent=JSON.stringify(DATA.provenance,null,2);
let first=Math.floor(DATA.frames.length*.75),index=first,playing=false,accum=0;
el('frame').min=0;el('frame').max=DATA.frames.length-1;
function draw() {
 const f=DATA.frames[index];
 for(const [layer,{group,meshes}] of Object.entries(layers)) {
  group.visible=el(layer).checked;
  for(const name of names) {const s=f[layer][name];meshes[name].visible=Boolean(s);if(s) SkeletonGeometry.place(meshes[name],v(s.origin),v(s.end));}
 }
 pointGroup.visible=el('points').checked;
 for(const name of pointNames) {const p=f.points[name];pointMeshes[name].visible=Boolean(p);if(p)pointMeshes[name].position.copy(v(p));}
 const selected=f[el('axisLayer').value][el('segment').value];
 axes.forEach((axis,k)=>{axis.visible=Boolean(selected&&el('axes').checked);if(selected){axis.position.copy(v(selected.origin));axis.setDirection(v(selected.axes[k]));}});
 el('clock').textContent=`Frame ${f.number} / sample ${index} / ${(f.time-DATA.frames[0].time).toFixed(3)} s`;
 el('status').textContent=JSON.stringify(f.status,null,2);el('frame').value=index;
}
function focus(shoulders=false) {
 const f=DATA.frames[index];let points=Object.values(f.points);
 if(shoulders)points=['left_acromion','right_acromion'].map(n=>f.points[n]).filter(Boolean);
 if(!points.length)return;
 const median=a=>a.sort((a,b)=>a-b)[Math.floor(a.length/2)];
 const center=shoulders?points.reduce((a,p)=>a.map((x,i)=>x+p[i]/points.length),[0,0,0]):[0,1,2].map(i=>median(points.map(p=>p[i])));
 const radius=shoulders?600:Math.max(900,median(points.map(p=>v(p).distanceTo(v(center))))*3);
 controls.target.copy(v(center));camera.position.copy(v(center)).add(new THREE.Vector3(radius,radius,radius*.4));controls.update();
}
el('play').onclick=()=>{playing=!playing;el('play').textContent=playing?'Pause':'Play';accum=0;};
el('frame').oninput=()=>{index=Number(el('frame').value);playing=false;el('play').textContent='Play';accum=0;draw();};
for(const id of ['independent','connected','fitted','points','axes','segment','axisLayer'])el(id).onchange=draw;
el('fit').onclick=()=>focus();el('shoulders').onclick=()=>focus(true);
el('late').onclick=()=>{first=Math.floor(DATA.frames.length*.75);index=first;accum=0;draw();focus();};
el('all').onclick=()=>{first=0;index=0;accum=0;draw();focus();};
new ResizeObserver(()=>{const box=el('scene').getBoundingClientRect();renderer.setSize(box.width,box.height);camera.aspect=box.width/box.height;camera.updateProjectionMatrix();}).observe(el('scene'));
let previous;
function duration(){return index<DATA.frames.length-1?DATA.frames[index+1].time-DATA.frames[index].time:index>0?DATA.frames[index].time-DATA.frames[index-1].time:Infinity;}
function animate(time){requestAnimationFrame(animate);const dt=previous==null?0:Math.min((time-previous)/1000,.25);previous=time;
 if(playing){accum+=dt;while(accum>=duration()){accum-=duration();index=index<DATA.frames.length-1?index+1:first;}draw();}
 controls.update();renderer.render(scene,camera);
}
draw();focus();requestAnimationFrame(animate);
