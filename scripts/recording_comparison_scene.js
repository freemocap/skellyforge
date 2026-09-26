/* Saved geometry only. No fitting, alignment, interpolation or rescaling of poses. */
function createComparisonScene(data, host, tooltip) {
  const scene=new THREE.Scene();scene.background=new THREE.Color('#101824');
  const camera=new THREE.PerspectiveCamera(45,1,1,20000);camera.up.set(0,0,1);
  const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));host.appendChild(renderer.domElement);
  const orbit=new THREE.OrbitControls(camera,renderer.domElement);
  const grid=new THREE.GridHelper(4000,40,0x587080,0x293b4a);grid.rotation.x=Math.PI/2;scene.add(grid);
  const sphere=new THREE.SphereGeometry(1,10,8),rod=new THREE.CylinderGeometry(.25,1,1,10);
  const vec=p=>new THREE.Vector3(...p),quaternion=q=>new THREE.Quaternion(q[1],q[2],q[3],q[0]);
  function material(color,wireframe=false){return new THREE.MeshBasicMaterial({color,wireframe,transparent:true,opacity:.85,depthWrite:false});}
  function dot(group,mat,radius,label){const m=new THREE.Mesh(sphere,mat);m.scale.setScalar(radius);m.userData.label=label;group.add(m);return m;}
  function stick(group,mat,label){const m=new THREE.Mesh(rod,mat);m.userData.label=label;group.add(m);return m;}
  function setRod(mesh,a,b,radius){const start=vec(a),end=vec(b),delta=end.clone().sub(start),length=delta.length();mesh.visible=length>1e-8;if(!mesh.visible)return;mesh.position.copy(start.add(end).multiplyScalar(.5));mesh.scale.set(radius,length,radius);mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),delta.normalize());}
  const solutions=data.solutions.map(solution=>{
    const group=new THREE.Group();scene.add(group);
    const solid=material(solution.color),wire=material(solution.color,true);
    const bodies=solution.definitions.map(def=>{
      const g=new THREE.Group();group.add(g);const hand=['Left hand','Right hand'].includes(def.region);
      const points=def.landmark_names.map(name=>dot(g,wire,hand?2.5:5,`${solution.label}\nLandmark: ${name}`));
      const bones=(def.attachments||[]).map(a=>stick(g,solid,`${solution.label}\nSegment: ${def.id}\nDistal attachment: ${a.label}`));
      const axes=new THREE.AxesHelper(30);g.add(axes);
      const origin=dot(g,solid,hand?2:4,`${solution.label}\nSegment: ${def.id} / local XYZ origin`);
      const label=document.createElement('span');label.textContent=def.id;label.style.cssText='position:absolute;pointer-events:none;font:10px system-ui;padding:1px 3px;background:#101824bb;color:'+solution.color;host.appendChild(label);
      return {g,points,bones,axes,origin,label};
    });
    const links=(solution.frames[0].linkages||[]).map(link=>({child:link.child,mesh:stick(group,solid,`${solution.label}\nRelaxed linkage to ${solution.definitions[link.child].id}`),a:dot(group,wire,6,'Clavicle-side linkage attachment'),b:dot(group,solid,3,'Upper-arm-side linkage attachment')}));
    return {group,solid,wire,bodies,links};
  });
  const reference=new THREE.Group();scene.add(reference);
  const keyMat=material('#56cedb'),savedMat=material('#acb7c2',true),savedBoneMat=material('#8596a5');savedBoneMat.opacity=.3;
  const keyNames=[...new Set(data.contexts.flatMap(f=>Object.keys(f.keypoints)))];
  const landmarkNames=[...new Set(data.contexts.flatMap(f=>Object.keys(f.landmarks)))];
  const segmentNames=[...new Set(data.contexts.flatMap(f=>Object.keys(f.segments)))];
  const keypoints=keyNames.map(n=>dot(reference,keyMat,data.hand_keypoints.includes(n)?1.8:3.6,'Tracker keypoint: '+n));
  const savedLandmarks=landmarkNames.map(n=>dot(reference,savedMat,data.hand_landmarks.includes(n)?2.5:5,'Saved posthoc landmark: '+n));
  const savedSegments=segmentNames.map(n=>stick(reference,savedBoneMat,'Saved posthoc segment: '+n));
  const definitionByName=new Map(data.solutions[0].definitions.map(b=>[b.id,b]));
  const landmarkRegion=new Map(data.solutions[0].definitions.flatMap(b=>b.landmark_names.map(n=>[n,b.region])));
  const keyRegions=new Map();for(const [landmark,source] of Object.entries(data.solutions[0].settings.direct_mapping_sources)){if(!keyRegions.has(source))keyRegions.set(source,new Set());keyRegions.get(source).add(landmarkRegion.get(landmark));}
  function draw(index,state){
    tooltip.hidden=true;grid.visible=state.grid;
    const inRegion=region=>state.region==='all'||region===state.region;
    data.solutions.forEach((solution,s)=>{
      const view=solutions[s],setting=state.solutions[s],frame=solution.frames[index];
      view.group.visible=setting.enabled&&setting.opacity>0;view.solid.opacity=view.wire.opacity=setting.opacity;
      view.solid.color.set(setting.color);view.wire.color.set(setting.color);
      view.bodies.forEach((body,b)=>{
        const def=solution.definitions[b],pose=frame.bodies[b],q=quaternion(pose.quaternion),t=pose.translation;
        body.g.visible=inRegion(def.region);
        body.points.forEach((point,j)=>{point.visible=state.landmarks;point.position.copy(vec(pose.fitted[j]));});
        body.bones.forEach((bone,j)=>{const local=[...def.attachments[j].position];local[2]*=pose.axial_scale;const endpoint=vec(local).applyQuaternion(q).add(vec(t)).toArray();setRod(bone,t,endpoint,def.region.includes('hand')?1:2);bone.visible&&=state.segments;});
        body.axes.quaternion.copy(q);body.axes.position.copy(vec(t));body.axes.visible=state.axes;
        body.origin.position.copy(vec(t));body.origin.visible=state.axes;
        body.label.hidden=!(state.axes&&body.g.visible&&view.group.visible);body.label.style.color=setting.color;
      });
      view.links.forEach((link,j)=>{const value=frame.linkages[j],visible=state.linkages&&inRegion(solution.definitions[value.child].region);setRod(link.mesh,value.parent_point,value.child_point,1.4);link.mesh.visible&&=visible;link.a.visible=link.b.visible=visible;link.a.position.copy(vec(value.parent_point));link.b.position.copy(vec(value.child_point));const label=`${solution.label}\nRelaxed linkage: ${solution.definitions[value.child].id}\nSeparation: ${Math.hypot(...value.local_displacement).toFixed(2)} mm`;link.mesh.userData.label=label;link.a.userData.label=label+'\nClavicle-side attachment';link.b.userData.label=label+'\nUpper-arm-side attachment';});
    });
    const context=data.contexts[index];
    keypoints.forEach((p,i)=>{const value=context.keypoints[keyNames[i]];p.visible=state.keypoints&&!!value&&(state.region==='all'||keyRegions.get(keyNames[i])?.has(state.region));if(value)p.position.copy(vec(value));});
    savedLandmarks.forEach((p,i)=>{const value=context.landmarks[landmarkNames[i]];p.visible=state.savedLandmarks&&!!value&&inRegion(landmarkRegion.get(landmarkNames[i]));if(value)p.position.copy(vec(value));});
    savedSegments.forEach((p,i)=>{const value=context.segments[segmentNames[i]];p.visible=false;if(value){setRod(p,value.origin,value.end,1);p.visible&&=state.savedSegments&&inRegion(definitionByName.get(segmentNames[i])?.region);}});
    render();
  }
  function render(){orbit.update();const width=host.clientWidth,height=host.clientHeight;solutions.forEach(view=>view.bodies.forEach(body=>{if(body.label.hidden)return;const p=body.origin.position.clone().project(camera);body.label.style.left=(p.x+1)*width/2+6+'px';body.label.style.top=(1-p.y)*height/2+'px';body.label.style.visibility=p.z>=-1&&p.z<=1?'visible':'hidden';}));renderer.render(scene,camera);}
  function reset(){const points=data.contexts.flatMap(f=>Object.values(f.keypoints)).map(vec),bounds=new THREE.Box3().setFromPoints(points),center=bounds.getCenter(new THREE.Vector3()),size=bounds.getSize(new THREE.Vector3()),distance=Math.max(size.x,size.y,size.z,250)*.85;orbit.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(distance,-distance,distance*.6));grid.position.set(center.x,center.y,0);render();}
  new ResizeObserver(()=>{camera.aspect=host.clientWidth/Math.max(host.clientHeight,1);camera.updateProjectionMatrix();renderer.setSize(host.clientWidth,host.clientHeight);render();}).observe(host);
  const ray=new THREE.Raycaster(),pointer=new THREE.Vector2();
  renderer.domElement.addEventListener('pointermove',event=>{const rect=renderer.domElement.getBoundingClientRect();pointer.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);ray.setFromCamera(pointer,camera);const candidates=[];scene.traverseVisible(o=>{if(o.userData.label&&o.material?.opacity>0)candidates.push(o);});const hit=ray.intersectObjects(candidates,false)[0];tooltip.hidden=!hit;if(hit){const isPoint=hit.object.geometry===sphere;tooltip.textContent=hit.object.userData.label+(isPoint?'\nPoint XYZ: ':'\nSurface XYZ: ')+(isPoint?hit.object.position:hit.point).toArray().map(v=>v.toFixed(1)).join(', ')+' mm';tooltip.style.left=Math.min(event.clientX+14,innerWidth-355)+'px';tooltip.style.top=Math.max(5,Math.min(event.clientY+12,innerHeight-160))+'px';}});
  renderer.domElement.addEventListener('pointerleave',()=>tooltip.hidden=true);
  reset();return {draw,render,reset,scene,solutions};
}
