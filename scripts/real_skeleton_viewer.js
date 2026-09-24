const el=id=>document.getElementById(id);
const scene=new THREE.Scene();scene.background=new THREE.Color(0x1a1a2e);
const camera=new THREE.PerspectiveCamera(50,1,1,100000);camera.up.set(0,0,1);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));el('scene').appendChild(renderer.domElement);
const controls=new THREE.OrbitControls(camera,renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff,.7));const light=new THREE.DirectionalLight(0xffffff,.5);light.position.set(1,1,1);scene.add(light);
const v=p=>new THREE.Vector3(...p);
const names=Object.keys(DATA.lengths),layers={};
const landmarkRadius=13;
for(const [name,color] of Object.entries({saved:0xffa860})) {
  const group=new THREE.Group(), meshes={};scene.add(group);
  for(const n of names) {const mesh=SkeletonGeometry.cylinder(DATA.lengths[n],landmarkRadius/2,color,1);mesh.userData.label=`Segment: ${n} (${name})`;group.add(mesh);meshes[n]=mesh;}
  layers[name]={group,meshes};
}
const pointLayers={};
for(const [field,label,color,radius,wireframe] of [
 ['points','Forge landmark',0xff55df,landmarkRadius,true],
 ['keypoints','Tracker keypoint',0x91ff51,8,false]]) {
 const group=new THREE.Group();scene.add(group);
 const geometry=new THREE.SphereGeometry(radius,12,8);
 const material=new THREE.MeshBasicMaterial({color,wireframe});
 const meshes={};
 for(const name of new Set(DATA.frames.flatMap(f=>Object.keys(f[field])))) {
  const mesh=new THREE.Mesh(geometry,material);mesh.userData.label=`${label}: ${name}`;
  group.add(mesh);meshes[name]=mesh;
 }
 pointLayers[field]={group,meshes};
}
const tooltip=el('tooltip'),raycaster=new THREE.Raycaster(),pointer=new THREE.Vector2();
let hovering=false;
renderer.domElement.addEventListener('pointermove',event=>{
 const rect=renderer.domElement.getBoundingClientRect();
 pointer.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);
 tooltip.style.left=`${Math.min(event.clientX+14,innerWidth-340)}px`;
 tooltip.style.top=`${Math.min(event.clientY+14,innerHeight-120)}px`;hovering=true;
});
renderer.domElement.addEventListener('pointerleave',()=>{hovering=false;tooltip.hidden=true;});
function hoverLabels() {
 if(!hovering)return;
 scene.updateMatrixWorld(true);raycaster.setFromCamera(pointer,camera);
 const visible=Object.values({...layers,...pointLayers}).flatMap(({group,meshes})=>group.visible?Object.values(meshes).filter(m=>m.visible):[]);
 const labels=[...new Set(raycaster.intersectObjects(visible,false).map(h=>h.object.userData.label))];
 tooltip.hidden=!labels.length;tooltip.textContent=labels.join('\n');
}
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
 for(const [field,{group,meshes}] of Object.entries(pointLayers)) {
  group.visible=el(field).checked;
  for(const [name,mesh] of Object.entries(meshes)) {const p=f[field][name];mesh.visible=Boolean(p);if(p)mesh.position.copy(v(p));}
 }
 const selected=f.saved[el('segment').value];
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
for(const id of ['saved','points','keypoints','axes','segment'])el(id).onchange=draw;
el('fit').onclick=()=>focus();el('shoulders').onclick=()=>focus(true);
el('late').onclick=()=>{first=Math.floor(DATA.frames.length*.75);index=first;accum=0;draw();focus();};
el('all').onclick=()=>{first=0;index=0;accum=0;draw();focus();};
new ResizeObserver(()=>{const box=el('scene').getBoundingClientRect();renderer.setSize(box.width,box.height);camera.aspect=box.width/box.height;camera.updateProjectionMatrix();}).observe(el('scene'));
let previous;
function duration(){return index<DATA.frames.length-1?DATA.frames[index+1].time-DATA.frames[index].time:index>0?DATA.frames[index].time-DATA.frames[index-1].time:Infinity;}
function animate(time){requestAnimationFrame(animate);const dt=previous==null?0:Math.min((time-previous)/1000,.25);previous=time;
 if(playing){accum+=dt;while(accum>=duration()){accum-=duration();index=index<DATA.frames.length-1?index+1:first;}draw();}
 controls.update();renderer.render(scene,camera);hoverLabels();
}
draw();focus();requestAnimationFrame(animate);
