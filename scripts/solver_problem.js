// Saved runs predating explicit bound metadata used these fixed fractions.
const LEGACY_AXIAL_LENGTH_BOUND_FRACTIONS=[0.25,2.0];
// Assembly schematic for the experiment definitions, not Ceres runtime introspection.
// Every displayed parameter value comes from the saved solve. No fitting occurs here.
function drawCeresProblem(){
  const structure=current.problem||selectedMode.problem;
  const chain=structure.chain||false;
  const displacement=structure.displacement||false;
  const restPrior=structure.rest_prior||false;
  const axial=structure.axial_segments||[];
  const relaxed=structure.relaxed_linkage_children||[];
  const linePrior=selectedMode.settings.chest_line_prior;
  const freeLengths=selectedMode.settings.free_axial_lengths===true;
  const [minimumLengthFraction,maximumLengthFraction]=selectedMode.settings.axial_length_bound_fractions||LEGACY_AXIAL_LENGTH_BOUND_FRACTIONS;
  const pathTo=b=>{const path=[b];while(b>0){b=structure.parents?structure.parents[b-1]:b-1;path.unshift(b);}return path;};
  const connected=structure.connected,temporal=structure.temporal,acceleration=structure.acceleration;
  const frames=selectedMode.frames,n=frames.length;
  const parameters=[],residuals=[];
  const format=values=>values.map(v=>Number(v).toFixed(4)).join(', ');
  function parameter(id,label,values,quaternion,index,extra={}){
    parameters.push({id,label,values,quaternion,index,...extra});return id;
  }
  function landmark(body,index,q,t,b){
    const count=body.observed.filter(p=>p!==null).length;
    if(!count)return;
    residuals.push({index,bodyIndex:b,kind:chain?"chain":"landmark",bodyLabel:experiment.bodies[b].label,name:`${chain?"ChainLandmarkResidual":"LandmarkResidual"} × ${count}`,connections:chain?[t,...pathTo(b).map(j=>`q-${index}-${j}`),...pathTo(b).filter(j=>relaxed.includes(j)).map(j=>`linkage-${index}-${j}`),...pathTo(b).filter(j=>axial.includes(j)).map(j=>`length-${index}-${j}`),...(displacement&&b===2?[`displacement-${index}`]:[])]:[q,t],
      detail:`${experiment.bodies[b].label}; frame ${index}; fixed local/observed XYZ; 3D residual`,
      formula:chain?'r = scale * (chain-derived translation + R(q) local - observed)':connected?'r = scale * (R(q) (local - attachment) + joint - observed)':'r = scale × (R(q) local + translation - observed)'});
  }
  const indices=temporal?Array.from({length:Math.min(3,n)},(_,j)=>Math.max(0,Math.min(n-3,frameIndex-1))+j):[frameIndex];
  for(const index of indices){
    const frame=frames[index];let jointId;
    if(chain){
      jointId=parameter(`root-${index}`,"Root segment position",frame.root,false,index,{bodyIndex:0});
    }else if(connected){
      const body=frame.bodies[0],q=body.quaternion;
      const joint=vector(experiment.bodies[0].attachment).applyQuaternion(new THREE.Quaternion(q[1],q[2],q[3],q[0])).add(vector(body.translation)).toArray();
      jointId=parameter(`joint-${index}`,'Shared joint position',frame.joint||joint,false,index);
    }
    for(const b of axial){
      const reference=selectedMode.settings.axial_reference_lengths[b],id=parameter(`length-${index}-${b}`,experiment.bodies[b].label+' length', [frame.lengths[axial.indexOf(b)]],false,index,{bodyIndex:b,kind:'length',lower:minimumLengthFraction*reference,upper:freeLengths?Infinity:maximumLengthFraction*reference});
      if(!freeLengths)residuals.push({index,bodyIndex:b,kind:'length_prior',bodyLabel:experiment.bodies[b].label,name:'LengthPriorResidual',connections:[id],detail:`Reference ${reference} mm; shortening scale ${selectedMode.settings.length_prior_fraction*reference} mm; lengthening scale ${(selectedMode.settings.lengthening_prior_fraction??selectedMode.settings.length_prior_fraction)*reference} mm; 1 component`,formula:'r = sqrt(time weight) * (length - reference) / (length < reference ? shortening scale : lengthening scale)'});
    }
    if(linePrior?.enabled&&frame.chest_line){
      const b=linePrior.body_index,path=pathTo(b);
      residuals.push({index,bodyIndex:b,kind:'chest_line',bodyLabel:'chest_center',name:'ChainLandmarkLineResidual',
        connections:[`root-${index}`,...path.map(j=>`q-${index}-${j}`),...path.filter(j=>axial.includes(j)).map(j=>`length-${index}-${j}`)],
        detail:`3 components; near-line scale ${linePrior.distance_scale_mm} mm; extra anterior scale ${linePrior.anterior_scale_mm} mm. Fixed mapped hip/shoulder frame. Modeling preference, not another measurement.`,
        formula:'r = sqrt(time weight) * [lateral / near scale, anterior / near scale, max(0, anterior) / anterior scale]'});
    }
    for(const b of relaxed){
      const link=frame.linkages.find(l=>l.child===b);
      const id=parameter(`linkage-${index}-${b}`,experiment.bodies[b].label+' parent-local XYZ displacement',link.local_displacement,false,index,{bodyIndex:b,kind:'displacement'});
      residuals.push({index,bodyIndex:b,kind:'prior',bodyLabel:experiment.bodies[b].label,name:'LinkageDisplacementPriorResidual',connections:[id],detail:`3 components; parent local mm; scale ${selectedMode.settings.linkage_scale_mm} mm; no hard bounds`,formula:'r = sqrt(time weight) * local displacement / scale'});
    }
    if(restPrior)for(let b=1;b<experiment.bodies.length;b++){
      const parent=structure.parents[b-1];
      residuals.push({index,bodyIndex:b,kind:'pose_prior',bodyLabel:experiment.bodies[b].label,name:'RelativePoseResidual',connections:[`q-${index}-${parent}`,`q-${index}-${b}`],detail:`Frame ${index}; 3 components; rest wxyz ${selectedMode.settings.rest_relative_quaternions[b-1].join(', ')}; scale ${selectedMode.settings.rest_pose_scale_radians} rad`,formula:'r = sqrt(time weight) / scale * Log(conj(q_rest) * conj(q_parent) * q_child)'});
    }
    if(displacement){
      const id=parameter(`displacement-${index}`,'Second linkage displacement',[frame.displacement],false,index,{kind:'displacement',bound:selectedMode.settings.displacement_bound_mm});
      residuals.push({index,kind:'prior',bodyLabel:'Zero-displacement preference',name:'DisplacementPriorResidual',connections:[id],detail:`Frame ${index}; 1 component; scale ${selectedMode.settings.displacement_scale_mm} mm`,formula:'r = sqrt(time weight) * displacement / displacement scale'});
    }
    frame.bodies.forEach((body,b)=>{
      if(!body.quaternion)return;
      const qId=parameter(`q-${index}-${b}`,`${experiment.bodies[b].label} quaternion`,body.quaternion,true,index,{bodyIndex:b});
      const tId=jointId||parameter(`t-${index}-${b}`,`${experiment.bodies[b].label} translation`,body.translation,false,index,{bodyIndex:b});
      landmark(body,index,qId,tId,b);
    });
  }
  if(temporal){
    const s=selectedMode.settings;
    const pairs=acceleration?[indices]:indices.slice(1).map((v,i)=>[indices[i],v]);
    const families=connected?[
      {bodyIndex:0,name:'Translation',scale:s.linear_motion_scale,ids:ids=>ids.map(i=>`${chain?'root':'joint'}-${i}`),label:chain?'root position':'shared joint'},
      ...experiment.bodies.map((body,b)=>({bodyIndex:b,name:'Quaternion',scale:s.angular_motion_scale,ids:ids=>ids.map(i=>`q-${i}-${b}`),label:body.label}))
    ]:[
      {bodyIndex:0,name:'Translation',scale:s.linear_motion_scale,ids:ids=>ids.map(i=>`t-${i}-0`),label:'translation'},
      {name:'Quaternion',scale:s.angular_motion_scale,ids:ids=>ids.map(i=>`q-${i}-0`),label:'rotation'}
    ];
    for(const b of relaxed)families.push({bodyIndex:b,name:'Translation',scale:s.linkage_acceleration_scale_mm_s2,ids:ids=>ids.map(i=>`linkage-${i}-${b}`),label:experiment.bodies[b].label+' parent-local displacement'});
    if(displacement)families.push({name:"Displacement",scale:s.displacement_acceleration_scale,ids:ids=>ids.map(i=>`displacement-${i}`),label:"second linkage displacement"});
    if(!freeLengths)for(const b of axial)for(const ids of pairs)residuals.push({index:null,bodyIndex:b,kind:'length_motion',bodyLabel:experiment.bodies[b].label,name:'DisplacementAccelerationResidual (length)',connections:ids.map(i=>`length-${i}-${b}`),detail:`Scalar length acceleration; scale ${selectedMode.settings.length_acceleration_scale} mm/s^2; 1 component`,formula:'Interval length velocity difference / (scale * sqrt(midpoint dt))'});
    for(const ids of pairs)for(const family of families){
      residuals.push({index:null,bodyIndex:family.bodyIndex,kind:family.name==="Displacement"?"displacement_motion":family.name==="Quaternion"?"angular":"linear",bodyLabel:family.label,name:`${family.name}${acceleration?'Acceleration':'Motion'}Residual`,connections:family.ids(ids),
        detail:`${family.label}; frames ${ids.join(', ')}; ${family.name==="Displacement"?1:3} components; scale ${family.scale}`,
        formula:acceleration?'Interval velocity difference / (motion scale * sqrt(midpoint dt))':'Neighbor pose increment / (motion scale * sqrt(dt))'});
    }
  }
  const observedCount=frame=>frame.bodies.reduce((sum,b)=>sum+b.observed.filter(p=>p!==null).length,0);
  const active=current.bodies.filter(b=>b.quaternion).length;
  const blockCount=relaxed.length*n+(temporal?(chain?experiment.bodies.length+1:connected?3:2)*n+(displacement?n:0)+axial.length*n:connected?3:2*active);
  const ambient=relaxed.length*n*3+(temporal?(chain?experiment.bodies.length*4+3:connected?11:7)*n+(displacement?n:0)+axial.length*n:connected?11:7*active);
  const tangent=relaxed.length*n*3+(temporal?(chain?(experiment.bodies.length+1)*3:connected?9:6)*n+(displacement?n:0)+axial.length*n:connected?9:6*active);
  const residualCount=relaxed.length*(2*n-2)+(temporal?frames.reduce((sum,f)=>sum+observedCount(f),0)+(chain?experiment.bodies.length+1:connected?3:2)*(n-(acceleration?2:1))+(displacement?2*n-2:0)+(restPrior?(experiment.bodies.length-1)*n:0)+(freeLengths?0:axial.length*(2*n-2))+(linePrior?.enabled?frames.filter(f=>f.chest_line).length:0):observedCount(current));
  el('problem-summary').textContent=`${temporal?'One sequence Problem':connected?'One connected Problem per frame':`${active} independent Problem(s) per frame`} | ${blockCount} parameter blocks (${ambient} stored values / ${tangent} tangent dimensions) | ${residualCount} residual blocks; ${displacement||axial.length?"geometry/pose residuals have 3 components, length/displacement residuals have 1":"each has 3 components"}. ${temporal?'Diagram shows one local time window; blocks outside it are omitted.':'Diagram shows the selected frame.'}`;
  renderCeresMap({parameters,residuals,indices});
  el('problem-value').textContent=(relaxed.length?'Selected shoulder linkages: child translation = parent translation + R(parent q) (parent attachment + fitted local XYZ displacement) - R(child q) child attachment. Zero-displacement and local acceleration residuals penalize separation and changes in separation. All other attachments remain exact. ':'')+(axial.length?'Axial segments: local Z scales by length/reference; local X/Y stay fixed. Both attachment sides use deformed geometry. Rigid segment geometry does not change; explicitly relaxed linkage displacements are described above.':displacement?'First linkage exact. Second linkage has a bounded scalar displacement along the middle segment local Z axis. Rigid segment geometry stays fixed. Bounds restrict feasible values; prior and acceleration residuals are weighted preferences.':chain?'Exact chain: root position and world quaternions determine all segment translations by attachment coincidence. No separate joint-position blocks or attachment residual penalties.':connected?'Exact linkage: both landmark residual families reference the SAME joint-position parameter block. There is no attachment residual or stiffness weight. Joint XYZ uses the saved block value when available, otherwise the returned parent pose.':temporal?'Temporal coupling is inside this Ceres Problem. Smoothness residuals connect neighboring parameter blocks; it is a weighted preference, not an exact constraint.':'No temporal residuals. These Problems do not share parameter blocks across frames or segments.')+(freeLengths?' Free lengths: nonnegative, no upper bound, no length-prior or length-acceleration residuals.':'')+(temporal?' Temporal residual blocks couple frames inside this Problem; no post-fit smoothing.':'')+'\nLossFunction: nullptr (ordinary squared residuals). AutoDiff computes derivatives. ceres::Solve adjusts the parameter blocks.\nAssembly schematic follows the experiment code; this is not a live Ceres graph dump. Known synthetic truth is only used for evaluation.';
}
