/* Presentation only: consume existing saved viewer results without fitting or moving points. */
const $=id=>document.getElementById(id);
const scene=new THREE.Scene();scene.background=new THREE.Color('#101923');
const camera=new THREE.PerspectiveCamera(42,1,1,30000);camera.up.set(0,0,1);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));$('scene').appendChild(renderer.domElement);
const orbit=new THREE.OrbitControls(camera,renderer.domElement);
const grid=new THREE.GridHelper(4000,40,0x425264,0x23313f);grid.rotation.x=Math.PI/2;scene.add(grid);
const groups={};for(const name of ['fit','reference','keypoints','landmarks','fitAxes','referenceAxes']){groups[name]=new THREE.Group();scene.add(groups[name]);}
const sphere=new THREE.SphereGeometry(1,10,8),rod=new THREE.CylinderGeometry(.7,1,1,10);
const materials=new Map();
function material(color,wire=false,opacity=1){const key=[color,wire,opacity].join();if(!materials.has(key))materials.set(key,new THREE.MeshBasicMaterial({color,wireframe:wire,transparent:opacity<1,opacity}));return materials.get(key);}
const V=p=>new THREE.Vector3(...p);
const sideColor=name=>name.startsWith('left_')?'#f0757c':name.startsWith('right_')?'#70aafa':'#dce5ef';
const isHand=name=>/finger|thumb|index|middle|ring|pinky|carpal|metacarp|phalanx|scaphoid|lunate|triquetrum|pisiform|trapez|capitate|hamate/.test(name);
let data=null,synthetic=false,index=0,times=[],frameIds=[],videos=[],playing=false,startTime=0,startClock=0,loadVersion=0;
let sticks=[],points=[],axes=[],imageRequest=0,shownImage='';const imageCache=new Map();
function stick(layer,name,a,b,color,radius=3){const mesh=new THREE.Mesh(rod,material(color,false,layer==='reference'?.9:1));mesh.userData.label=(layer==='fit'?'Fitted segment: ':'Reference segment: ')+name;groups[layer].add(mesh);sticks.push({mesh,a,b,radius});}
function point(layer,name,value,color,wire=false){const mesh=new THREE.Mesh(sphere,material(color,wire));mesh.scale.setScalar(isHand(name)?3:6);mesh.userData.label=(layer==='keypoints'?'Keypoint: ':'Mapped landmark: ')+name;groups[layer].add(mesh);points.push({mesh,value});}
// Basis vectors are saved columns of the rotation matrix, or derived from the
// saved wxyz quaternion. Never infer roll from a stick's two endpoints.
function basisFromQuaternion(q){const rotation=new THREE.Quaternion(q[1],q[2],q[3],q[0]);return [[1,0,0],[0,1,0],[0,0,1]].map(a=>V(a).applyQuaternion(rotation).toArray());}
function segmentAxes(layer,name,origin,basis){
  const group=new THREE.Group();groups[layer].add(group);
  const label=(layer==='fitAxes'?'Fitted skeleton':'Reference segment')+': '+name;
  const marker=new THREE.Mesh(sphere,material(layer==='fitAxes'?'#eaf2ff':'#efb95d',layer==='referenceAxes'));
  marker.scale.setScalar(isHand(name)?2.5:5);marker.userData.label=label+' / local XYZ origin';group.add(marker);
  const length=isHand(name)?14:50;
  const arrows=[0xff5555,0x59d66f,0x599cff].map((color,j)=>{
    const arrow=new THREE.ArrowHelper(V([1,0,0]),V([0,0,0]),length,color,length*.2,length*.09);
    arrow.line.userData.label=arrow.cone.userData.label=label+' / '+['X','Y','Z'][j]+' axis';group.add(arrow);return arrow;
  });
  axes.push({group,origin,basis,marker,arrows,length});
}
function poseAxes(axis){const origin=axis.origin(index),basis=axis.basis(index);axis.group.visible=!!origin&&!!basis;if(!axis.group.visible)return;axis.group.position.copy(V(origin));axis.arrows.forEach((arrow,j)=>arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),V(basis[j]).normalize()));}
function fittedEndpoint(def,pose,attachment){const p=[...attachment.position];p[2]*=pose.axial_scale??1;const q=pose.quaternion;return V(p).applyQuaternion(new THREE.Quaternion(q[1],q[2],q[3],q[0])).add(V(pose.translation)).toArray();}
function construct(){
  for(const axis of axes)for(const arrow of axis.arrows){arrow.line.material.dispose();arrow.cone.material.dispose();}
  for(const group of Object.values(groups))group.clear();sticks=[];points=[];axes=[];
  if(synthetic){
    data.segments_meta.forEach((def,b)=>{
      segmentAxes('fitAxes',def.name,i=>data.frames[i].segments[b].connected_origin,i=>data.frames[i].segments[b].connected_basis);
      segmentAxes('referenceAxes',def.name,i=>data.frames[i].segments[b].origin,i=>data.frames[i].segments[b].basis);
      stick('fit',def.name,i=>data.frames[i].segments[b].connected_origin,i=>data.frames[i].segments[b].connected_end,sideColor(def.name),isHand(def.name)?1.4:3);
      stick('reference',def.name,i=>data.frames[i].segments[b].origin,i=>data.frames[i].segments[b].end,'#efb95d',isHand(def.name)?1.8:3.8);
    });
    data.landmarks_meta.forEach((name,p)=>point('landmarks',name,i=>data.frames[i].landmarks[p],'#f7d487',true));
  }else{
    const solution=data.solutions[0];
    solution.definitions.forEach((def,b)=>{for(const a of def.attachments||[])stick('fit',def.id+' / '+a.label,i=>solution.frames[i].bodies[b].translation,i=>fittedEndpoint(def,solution.frames[i].bodies[b],a),sideColor(def.id),isHand(def.id)?1.4:3);});
    solution.definitions.forEach((def,b)=>segmentAxes('fitAxes',def.id,i=>solution.frames[i].bodies[b].translation,i=>basisFromQuaternion(solution.frames[i].bodies[b].quaternion)));
    const segmentNames=[...new Set(data.contexts.flatMap(c=>Object.keys(c.segments)))];
    segmentNames.forEach(name=>stick('reference',name,i=>data.contexts[i].segments[name]?.origin,i=>data.contexts[i].segments[name]?.end,'#efb95d',isHand(name)?1.8:3.8));
    segmentNames.forEach(name=>segmentAxes('referenceAxes',name,i=>data.contexts[i].segments[name]?.origin,i=>data.contexts[i].segments[name]?.axes));
    const keyNames=[...new Set(data.contexts.flatMap(c=>Object.keys(c.keypoints)))];
    keyNames.forEach(name=>point('keypoints',name,i=>data.contexts[i].keypoints[name],'#6edacc'));
    const landmarkNames=[...new Set(data.contexts.flatMap(c=>Object.keys(c.landmarks)))];
    landmarkNames.forEach(name=>point('landmarks',name,i=>data.contexts[i].landmarks[name],'#f7d487',true));
  }
}
function poseStick(item){const a=item.a(index),b=item.b(index);item.mesh.visible=!!a&&!!b;if(!item.mesh.visible)return;const start=V(a),delta=V(b).sub(start),length=delta.length();item.mesh.visible=length>1e-8;if(!item.mesh.visible)return;item.mesh.position.copy(start.add(V(b)).multiplyScalar(.5));item.mesh.scale.set(item.radius,length,item.radius);item.mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),delta.normalize());}
function draw(){if(!data)return;sticks.forEach(poseStick);axes.forEach(poseAxes);points.forEach(p=>{const v=p.value(index);p.mesh.visible=!!v;if(v)p.mesh.position.copy(V(v));});for(const name of Object.keys(groups))groups[name].visible=$(name).checked;renderer.render(scene,camera);}
function reset(){if(!data)return;const bounds=new THREE.Box3();for(const item of sticks){const a=item.a(index),b=item.b(index);if(a)bounds.expandByPoint(V(a));if(b)bounds.expandByPoint(V(b));}if(bounds.isEmpty())return;const center=bounds.getCenter(new THREE.Vector3()),size=bounds.getSize(new THREE.Vector3());const distance=Math.max(size.x,size.y,size.z,400)*1.35;orbit.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(distance*.2,-distance,distance*.28));grid.position.set(center.x,center.y,0);orbit.update();draw();}
function layout(){const show=$('video').checked&&videos.length>0;$('video-panel').hidden=$('divider').hidden=!show;}
function decodedImage(url){if(imageCache.has(url))return imageCache.get(url);const image=new Image();const promise=new Promise((resolve,reject)=>{image.onload=()=>{(image.decode?image.decode():Promise.resolve()).then(()=>resolve(image),reject);};image.onerror=()=>reject(Error('Annotated frame unavailable'));});imageCache.set(url,promise);image.src=url;promise.catch(()=>{});while(imageCache.size>40)imageCache.delete(imageCache.keys().next().value);return promise;}
function showVideo(){const request=++imageRequest;if(!$('video').checked||!videos.length)return;const view=videos[Number($('camera').value)],frame=frameIds[index],url='/'+view.base+String(frame).padStart(6,'0')+'.jpg?v='+view.sha256.slice(0,12);if(shownImage===url)return;$('video-note').textContent='Loading frame '+frame;decodedImage(url).then(image=>{if(request!==imageRequest)return;$('video-image').src=image.src;$('video-image').hidden=false;shownImage=url;$('video-note').textContent='Annotated / frame '+frame;},()=>{if(request===imageRequest)$('video-note').textContent='Frame '+frame+' unavailable';});}
function refresh(){if(!data)return;draw();$('timeline').value=index;$('frame').textContent='Frame '+frameIds[index]+' / '+frameIds.at(-1)+'  |  '+times[index].toFixed(2)+' s';showVideo();}
function pause(){playing=false;$('play').textContent='Play';}
function seek(next){pause();index=Math.max(0,Math.min(times.length-1,next));refresh();}
async function load(name){const version=++loadVersion;pause();data=null;$('loading').hidden=false;$('loading').textContent='Loading '+name+'...';for(const group of Object.values(groups))group.clear();renderer.render(scene,camera);imageRequest++;shownImage='';$('video-image').hidden=true;imageCache.clear();$('tooltip').hidden=true;
  try{const response=await fetch('/review-data/'+name);if(!response.ok)throw Error(await response.text());const incoming=await response.json();if(version!==loadVersion)return;data=incoming;synthetic=name==='synthetic';index=0;times=synthetic?data.frames.map((_,i)=>i/data.fps):data.times;frameIds=synthetic?data.frames.map((_,i)=>i):data.frame_ids;videos=synthetic?[]:data.videos;
    $('camera').replaceChildren(...videos.map((v,i)=>new Option('Camera '+(i+1),String(i))));$('camera').title=videos[0]?.label||'';$('video').disabled=!videos.length;$('keypoints').disabled=synthetic;
    $('reference').title=synthetic?'Hydrated segments before Ceres fitting':'Saved post-hoc segments before Ceres fitting';
    $('timeline').max=times.length-1;$('caption').textContent=synthetic?'Known synthetic motion / Ceres fit':'Prepared recording / Ceres fit';construct();layout();refresh();reset();$('loading').hidden=true;
  }catch(error){if(version!==loadVersion)return;$('loading').textContent=error.message;$('frame').textContent='';}
}
$('dataset').onchange=()=>load($('dataset').value);for(const name of Object.keys(groups))$(name).onchange=draw;$('video').onchange=()=>{layout();showVideo();};$('camera').onchange=()=>{$('camera').title=videos[Number($('camera').value)]?.label||'';showVideo();};$('reset').onclick=reset;
$('timeline').oninput=()=>seek(Number($('timeline').value));$('previous').onclick=()=>seek(index-1);$('next').onclick=()=>seek(index+1);
$('play').onclick=()=>{if(!data)return;if(playing){pause();return;}if(index===times.length-1)index=0;startTime=times[index];startClock=performance.now();playing=true;$('play').textContent='Pause';refresh();};$('speed').onchange=()=>{if(playing){startTime=times[index];startClock=performance.now();}};
function animate(now){if(playing&&data){const previous=index;let t=startTime+(now-startClock)/1000*Number($('speed').value);if(t>times.at(-1)){if($('loop').checked){t=times[0];startTime=t;startClock=now;index=0;}else{index=times.length-1;pause();}}let next=index;while(next+1<times.length&&times[next+1]<=t)next++;index=next;if(index!==previous)refresh();}orbit.update();renderer.render(scene,camera);requestAnimationFrame(animate);}requestAnimationFrame(animate);
new ResizeObserver(()=>{const host=$('scene');camera.aspect=host.clientWidth/Math.max(host.clientHeight,1);camera.updateProjectionMatrix();renderer.setSize(host.clientWidth,host.clientHeight);}).observe($('scene'));
const divider=$('divider');function resizeVideo(width){const value=Math.max(100,Math.min(innerWidth-250,width));document.documentElement.style.setProperty('--video-width',value+'px');divider.setAttribute('aria-valuenow',value);}
divider.onpointerdown=e=>{divider.setPointerCapture(e.pointerId);document.body.classList.add('resizing');divider.onpointermove=p=>resizeVideo(p.clientX);};divider.onpointerup=divider.onpointercancel=()=>{divider.onpointermove=null;document.body.classList.remove('resizing');};divider.onkeydown=e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();resizeVideo($('video-panel').clientWidth+(e.key==='ArrowLeft'?-20:20));}};
const ray=new THREE.Raycaster();ray.params.Line.threshold=3;const mouse=new THREE.Vector2();renderer.domElement.onpointermove=e=>{const r=renderer.domElement.getBoundingClientRect();mouse.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);ray.setFromCamera(mouse,camera);const visible=[];scene.traverseVisible(o=>{if(o.userData.label)visible.push(o);});const hit=ray.intersectObjects(visible,false)[0];$('tooltip').hidden=!hit;if(hit){$('tooltip').textContent=hit.object.userData.label;$('tooltip').style.left=Math.min(e.clientX+12,innerWidth-365)+'px';$('tooltip').style.top=Math.min(e.clientY+12,innerHeight-60)+'px';}};renderer.domElement.onpointerleave=()=>$('tooltip').hidden=true;
document.addEventListener('keydown',e=>{if(/INPUT|SELECT|BUTTON/.test(e.target.tagName))return;if(e.code==='Space'){e.preventDefault();$('play').click();}if(e.key==='ArrowLeft')seek(index-1);if(e.key==='ArrowRight')seek(index+1);});
load('synthetic');
