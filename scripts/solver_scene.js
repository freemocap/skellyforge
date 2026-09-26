// Shared rendering: only saved geometry is drawn. No fitting or pose adjustments.
const el=id=>document.getElementById(id);
const scene=new THREE.Scene();scene.background=new THREE.Color(0x151b29);
const groundGrid=new THREE.GridHelper(2000,20,0x617b94,0x304156);
groundGrid.rotation.x=Math.PI/2;groundGrid.material.transparent=true;groundGrid.material.opacity=.65;scene.add(groundGrid);
const camera=new THREE.PerspectiveCamera(45,1,1,10000);camera.up.set(0,0,1);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));el('scene').appendChild(renderer.domElement);
const controls=new THREE.OrbitControls(camera,renderer.domElement);
const groups={};for(const name of ['truth','initial','fitted','observed','residuals','axes','contextSegments','contextLandmarks','contextKeypoints','chestCenterline']){groups[name]=new THREE.Group();scene.add(groups[name]);}
const vector=p=>new THREE.Vector3(...p);
function line(group,a,b,color,label){const mesh=new THREE.Line(new THREE.BufferGeometry().setFromPoints([vector(a),vector(b)]),new THREE.LineBasicMaterial({color}));mesh.userData.label=label;group.add(mesh);}
function point(group,p,color,label,wire=false,radius=wire?5:3.5){const mesh=new THREE.Mesh(new THREE.SphereGeometry(radius,12,8),new THREE.MeshBasicMaterial({color,wireframe:wire}));mesh.position.copy(vector(p));mesh.userData.label=label+'\n'+p.map(x=>x.toFixed(2)).join(', ')+' mm';group.add(mesh);}
function segmentRod(group,a,b,color,label,wire=false,emphasized=false){
  const start=vector(a),end=vector(b),delta=end.clone().sub(start);
  if(delta.length()<1e-8)return;
  const mesh=new THREE.Mesh(new THREE.CylinderGeometry(emphasized?1.2:(wire?2.2:1.6),emphasized?2.8:(wire?2.2:1.6),delta.length(),12),new THREE.MeshBasicMaterial({color,wireframe:wire}));
  mesh.position.copy(start.add(end).multiplyScalar(.5));
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),delta.normalize());
  mesh.userData.label=label;group.add(mesh);
}
function clear(){for(const group of Object.values(groups))for(const child of [...group.children]){child.traverse(o=>{o.geometry?.dispose();if(o.material)for(const m of Array.isArray(o.material)?o.material:[o.material])m.dispose();});group.remove(child);}}
const axisLabels=[];
function updateAxisLabels(){
  const host=el('scene');
  for(const {node,position} of axisLabels){
    const p=vector(position).project(camera);
    node.hidden=!groups.axes.visible||p.z < -1||p.z > 1||Math.abs(p.x)>1||Math.abs(p.y)>1;
    node.style.left=((p.x+1)*host.clientWidth/2+7)+'px';
    node.style.top=((-p.y+1)*host.clientHeight/2+7)+'px';
  }
}
function segmentStyle(body,index){
  const axial=body.mechanical_model==='axial'||(selectedMode.problem?.axial_segments||[]).includes(index);
  const model=axial?'Variable local-Z length; local X/Y fixed':'Rigid local geometry';
  const count=body.target_count??body.observed.filter(Boolean).length;
  const rank=body.target_rank;
  const supported=!axial&&rank>=2;
  const support=rank===undefined?`${count} supplied landmark targets; pose observability not classified`:
    supported?`${count} non-collinear mapped targets can determine this rigid pose locally`:
    `${count} mapped targets; this segment's pose is not fully determined by its own targets`;
  const palette=[0xffbe63,0x78d6a3,0x91a9ff,0xe692d4,0x79d9e8];
  return {color:el('segment-colors').value==='support'?(supported?0x63d6a3:count?0xd9a1ff:0x9aabba):
    el('segment-colors').value==='model'?(axial?0xffbe63:0x79b7ff):palette[index%5],label:model+'\n'+support};
}
function drawScene(frame,experiment){
  clear();
  axisLabels.length=0;el('axis-labels').replaceChildren();
  el('segment-legend').textContent=el('segment-colors').value==='support'?
    'Green: rigid pose locally determined by non-collinear mapped targets. Purple: insufficient own targets. Gray: no direct targets. Connections and residual preferences still couple all segments; this is not a global observability test.':
    el('segment-colors').value==='model'?'Blue: rigid geometry. Amber: variable axial length. Connected joints coincide exactly; rest-pose and temporal residuals are weighted preferences.':'Different segment colors; hover for geometry and target support.';
  frame.bodies.forEach((body,index)=>{
    const definition=experiment.bodies[index];
    const style=segmentStyle(body,index),segmentColor=style.color;
    for(const [name,color,qKey,tKey] of [['truth',0x42d9eb,'reference_quaternion','reference_translation'],['initial',0x8a96ab,'initial_quaternion','initial_translation'],['fitted',0xffac54,'quaternion','translation']]){
      if(!body[qKey])continue;
      const points=body[name]||body.local;
      for(const [i,j] of definition.edges)line(groups[name],points[i],points[j],color,`${definition.label} / ${name}`);
      points.forEach((p,i)=>{const withheld=!body.observed[i];point(groups[name],p,withheld&&name==='fitted'?0xc99aff:color,`${name}: ${definition.landmark_names[i]}${withheld?' (no direct target)':''}`,name!=='fitted',name==='truth'?5:((frame.recording_context&&!withheld)?5:(withheld?3.5:2.5)));});
      const q=body[qKey],t=body[tKey];
      const axes=new THREE.AxesHelper(35);axes.quaternion.set(q[1],q[2],q[3],q[0]);axes.position.copy(vector(t));
      if(name==='fitted'&&(el('body-region').value==='all'||definition.region===el('body-region').value)&&(el('body-focus').value==='all'||Number(el('body-focus').value)===index)){
        axes.userData.label=definition.label+' fitted local XYZ frame\n'+style.label;groups.axes.add(axes);
        point(groups.axes,t,segmentColor,definition.label+' / fitted axis origin\n'+style.label,false,4);
        const node=document.createElement('span');node.className='axis-origin-label';node.textContent=definition.label;
        el('axis-labels').appendChild(node);axisLabels.push({node,position:t});
      }
      for(const attachment of definition.attachments||(definition.attachment?[{position:definition.attachment,label:"attachment"}]:[])){
        const attachmentPoint=[...attachment.position];if(name==='fitted')attachmentPoint[2]*=body.axial_scale||1;if(name==='truth')attachmentPoint[2]*=body.reference_axial_scale||1;
        const joint=vector(attachmentPoint).applyQuaternion(axes.quaternion).add(vector(t));
        const rodColor=name==='initial'?0x8a96ab:segmentColor;
        const marker=new THREE.Mesh(new THREE.SphereGeometry(name==='truth'?5.5:4,12,8),new THREE.MeshBasicMaterial({color:rodColor,wireframe:name!=='fitted'}));
        marker.position.copy(joint);marker.userData.label=`${definition.label} / ${name} ${attachment.label} (attachment, not landmark)`;groups[name].add(marker);
        segmentRod(groups[name],t,joint.toArray(),rodColor,`${definition.label} / ${name}\n${style.label}\nEndpoint: ${attachment.label}`,name!=='fitted',!!frame.recording_context&&name==='fitted');
      }
    }
    body.observed.forEach((p,i)=>{if(!p)return;point(groups.observed,p,0xff64b4,`Observed target: ${definition.landmark_names[i]}`,!!frame.recording_context,frame.recording_context?8:3.5);line(groups.residuals,body.fitted[i],p,0xff64b4,`${definition.landmark_names[i]}: fitted-to-observed error`);});
  });
  if(experiment.displacement_link){
    const link=experiment.displacement_link;
    for(const [name,qKey,tKey,color] of [['truth','reference_quaternion','reference_translation',0x42d9eb],['fitted','quaternion','translation',0xf2d17b],['initial','initial_quaternion','initial_translation',0x8a96ab]]){
      const a=frame.bodies[link.parent],b=frame.bodies[link.child];
      if(!a[qKey]||!b[qKey])continue;
      const transform=(point,body)=>{const q=body[qKey];return vector(point).applyQuaternion(new THREE.Quaternion(q[1],q[2],q[3],q[0])).add(vector(body[tKey])).toArray();};
      line(groups[name],transform(link.parent_attachment,a),transform(link.child_attachment,b),color,`${name}: modeled linkage displacement`);
    }
  }
  if(frame.chest_line){
    const c=frame.chest_line,g=groups.chestCenterline;
    line(g,c.origin,c.shoulder,0xf6e58d,'Mapped hip-center / shoulder-center line (geometric preference, not independent measurement)');
    point(g,c.origin,0xf6e58d,'Mapped hip center / centerline origin',true,6);
    point(g,c.shoulder,0xf6e58d,'Mapped shoulder center',true,6);
    point(g,c.projection,0xffffff,'Closest point on centerline to fitted chest_center',true,7);
    point(g,c.fitted,0xff64b4,'Fitted chest_center landmark',false,5);
    line(g,c.projection,c.fitted,0xff64b4,`Chest-center displacement: lateral ${c.lateral_mm.toFixed(2)} mm; anterior ${c.anterior_mm.toFixed(2)} mm (negative = posterior)`);
    line(g,c.projection,vector(c.projection).addScaledVector(vector(c.anterior),80).toArray(),0x69d488,'Centerline frame: anterior (+); opposite is posterior');
  }
  if(frame.recording_context){
    const context=frame.recording_context;
    for(const [name,segment] of Object.entries(context.segments)){
      line(groups.contextSegments,segment.origin,segment.end,0x859baa,`Saved posthoc segment: ${name} (context, not Ceres fit)`);
      const material=groups.contextSegments.children.at(-1).material;material.transparent=true;material.opacity=.4;
    }
    for(const [name,p] of Object.entries(context.landmarks))point(groups.contextLandmarks,p,0xaec8d8,`Saved landmark: ${name}`,true,3.5);
    for(const [name,p] of Object.entries(context.keypoints))point(groups.contextKeypoints,p,0x65cee0,`Saved tracker keypoint: ${name}`,false,2);
  }
  for(const [key,g] of Object.entries(groups))g.visible=el(key).checked;
  updateAxisLabels();
}
function reset(){
  if(experiment?.metadata?.recording&&selectedMode){
    const points=selectedMode.frames.flatMap(f=>[...f.bodies.flatMap(b=>b.fitted),...Object.values(f.recording_context?.keypoints||{})]).map(vector);
    const bounds=new THREE.Box3().setFromPoints(points),center=bounds.getCenter(new THREE.Vector3());
    const size=bounds.getSize(new THREE.Vector3()),distance=Math.max(size.x,size.y,size.z,250)*1.4;
    controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(distance,-distance,distance*.65));
  }else{camera.position.set(650,-750,550);controls.target.set(40,20,180);}
  controls.update();
}
function setupScene(){
  el('grid').addEventListener('change',()=>groundGrid.visible=el('grid').checked);
  const ray=new THREE.Raycaster(),pointer=new THREE.Vector2();
  renderer.domElement.addEventListener('pointermove',e=>{const r=renderer.domElement.getBoundingClientRect();pointer.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);ray.setFromCamera(pointer,camera);const hit=ray.intersectObjects(Object.values(groups).filter(g=>g.visible).flatMap(g=>g.children),true).find(h=>h.object.userData.label);el('tooltip').hidden=!hit;if(hit){el('tooltip').textContent=hit.object.userData.label;el('tooltip').style.left=Math.min(e.clientX+12,innerWidth-250)+'px';el('tooltip').style.top=e.clientY+12+'px';}});
  renderer.domElement.addEventListener('pointerleave',()=>el('tooltip').hidden=true);
  new ResizeObserver(()=>{const box=el('scene');camera.aspect=box.clientWidth/Math.max(box.clientHeight,1);camera.updateProjectionMatrix();renderer.setSize(box.clientWidth,box.clientHeight);}).observe(el('scene'));
}
