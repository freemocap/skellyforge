/* Audit table: describe saved solver settings, not inferred anatomy. */
function describeComparisonFit(solution){
  const s=solution.settings,g=s.shoulder_geometry;
  const free=s.free_axial_lengths===true,short=s.length_prior_fraction,long=s.lengthening_prior_fraction??short;
  return {
    spine:solution.processing?.fixed_length_segments?.length?'Fixed length: '+solution.processing.fixed_length_segments.map(i=>solution.definitions[i].id).join(', '):s.shared_axial_length?'One shared length':s.length_proportion_prior?.enabled?'Three flexible / ratio prior':s.length_equality_prior?.enabled?'Free + equal-length prior':free?'Free lengths':'Length priors',
    spineDetail:s.shared_axial_length?`Exact lumbar : thoracic : cervical = ${s.shared_axial_length.ratios.join(' : ')}`:s.length_proportion_prior?.enabled?`Lumbar : thoracic : cervical = ${s.length_proportion_prior.ratios.join(' : ')}; scale ${s.length_proportion_prior.scale_mm} mm`:s.length_equality_prior?.enabled?`Difference scale ${s.length_equality_prior.scale_mm} mm`:free?(s.chest_line_prior?.enabled?'Centerline preference':'No centerline preference'):`Prior scales ${short*100}% / ${long*100}%`,
    offsets:!g?'Original':g.forward_factor<1?'Lower + closer':'Lowered',
    shoulders:s.relaxed_linkage_children?.length?'Relaxed':'Exact',
    shoulderDetail:s.relaxed_linkage_children?.length?`${s.linkage_scale_mm} mm residual scale`:'Coincident attachments',
    rms:solution.summary['Observed landmark RMS (mm)'],
  };
}
function createComparisonTable(data,state,refresh){
  const benchmark=data.solutions.some(s=>s.benchmark_timing);
  const rows=[];let selected=data.solutions.findIndex(s=>s.id==='sc_75_rest_25');if(selected<0)selected=data.solutions.findIndex(s=>s.id==='proportional_spine');if(selected<0)selected=data.solutions.findIndex(s=>s.id==='equal_spine_lengths');if(selected<0)selected=data.solutions.findIndex(s=>s.id==='lower_sc_relaxed');if(selected<0)selected=0;
  const table=node('table');table.className='fit-table';const caption=node('caption','Saved fits â€” identical recording frames, different solver settings');caption.className='sr-only';table.appendChild(caption);
  const head=node('thead'),headRow=node('tr');
  for(const text of (benchmark?['Display','Option','Rest','Bounds','Twist','Status','Ceres','Wall','RMS','Actions']:['Display','Fit','Spine','SC depth','Rest','Shoulders','Status','Time','RMS','Actions'])){const th=node('th',text);th.setAttribute('scope','col');headRow.appendChild(th);}
  head.appendChild(headRow);table.appendChild(head);const body=node('tbody');table.appendChild(body);byId('solutions').appendChild(table);
  function cell(row,primary,secondary){const td=node('td');td.appendChild(node('div',primary));if(secondary)td.appendChild(node('div',secondary,'cell-detail'));row.appendChild(td);return td;}
  data.solutions.forEach((solution,index)=>{
    const description=describeComparisonFit(solution),row=node('tr');row.setAttribute('aria-label',solution.label);row.style.setProperty('--color',solution.color);
    const display=node('td'),displayControls=node('div',undefined,'row-display');
    const checkbox=node('input');checkbox.type='checkbox';checkbox.setAttribute('aria-label','Overlay '+solution.label);displayControls.appendChild(checkbox);
    const color=node('input');color.type='color';color.setAttribute('aria-label',solution.label+' color');displayControls.appendChild(color);
    const opacity=node('input');opacity.type='number';opacity.min=0;opacity.max=100;opacity.step=5;opacity.setAttribute('aria-label',solution.label+' opacity percent');opacity.title='Opacity (%)';displayControls.appendChild(opacity);display.appendChild(displayControls);
    const solo=node('button','Solo');solo.setAttribute('aria-label','Show only '+solution.label);solo.onclick=()=>{state.solutions.forEach((s,i)=>s.enabled=i===index);refresh();};row.appendChild(display);
    const processing=solution.processing;
    if(benchmark){
      const option=cell(row,solution.label);option.title=solution.objective;
      cell(row,`${100*solution.settings.length_prior_fraction}%`);
      const bounds=solution.settings.shared_axial_length?.bound_fractions;
      cell(row,bounds?bounds.map(x=>Math.round(100*x)+'%').join(' to '):'None');
      cell(row,solution.settings.relative_twist_priors?.length?solution.settings.relative_twist_priors[0].scale+' rad':'Off');
    }else{
    cell(row,processing?`${processing.active_frames}-frame${processing.refined?' + global':''}${solution.id.startsWith('tolerance_')?' / '+processing.function_tolerance.toExponential()+(processing.initial_function_tolerance?' / strict start':''):''}`:'Full sequence');
    let spineLabel=solution.settings.shared_axial_length?'Shared '+solution.settings.shared_axial_length.ratios.join(':'):solution.settings.length_proportion_prior?.enabled?solution.settings.length_proportion_prior.ratios.join(':'):solution.settings.length_equality_prior?.enabled?'Equal prior':solution.settings.free_axial_lengths?'Free':'Length prior';
    if(solution.settings.segment_axis_prior)spineLabel+=' + axis';
    const spineCell=cell(row,spineLabel);spineCell.title=description.spineDetail;
    cell(row,solution.settings.shoulder_geometry?`${Math.round(100*solution.settings.shoulder_geometry.forward_factor)}%`:'Original');
    const restCell=cell(row,(!solution.settings.free_axial_lengths||solution.settings.free_length_rest_prior)?`${100*solution.settings.length_prior_fraction}%`:'Off');restCell.title='Rest-length residual scale as a fraction of reference length; smaller is stronger, not a hard bound.';
    cell(row,description.shoulders);
    }
    const axialAccepted=solution.summary?.['Axial position acceptance passed'];
    const numericalStatus=processing&&!processing.refined?`${processing.windows.filter(w=>w.converged).length}/${processing.windows.length}`:(solution.converged?'Converged':'Limit');
    const status=axialAccepted===false?'Position check failed':numericalStatus;
    const result=cell(row,status);result.title=axialAccepted===false?'One or more fitted axial landmarks exceeded the experimental displacement limit. Numerical convergence does not imply an acceptable fit.':processing&&!processing.refined?'Windows meeting a convergence criterion / total windows':solution.report;result.className=solution.converged&&axialAccepted!==false?'result-ok':'result-warning';const warning=node('span','','frame-warning');result.appendChild(warning);
    cell(row,solution.seconds.toFixed(1)+' s').className='numeric';
    if(benchmark){const seconds=solution.benchmark_timing?.window_processing_wall_seconds??processing?.wall_seconds;const wall=cell(row,Number.isFinite(seconds)?seconds.toFixed(1)+' s':'Unavailable');wall.title='Window controller plus Ceres and final evaluation. File loading and viewer preparation are reported in Details.';}
    const rms=cell(row,Number.isFinite(description.rms)?description.rms.toFixed(2)+' mm':'Unavailable');rms.className='numeric';
    const inspect=node('button','Details');inspect.setAttribute('aria-label','Inspect '+solution.label);inspect.onclick=()=>{selected=index;updateInspector();byId('fit-inspector').open=true;byId('fit-inspector').scrollIntoView?.({block:'nearest'});};const inspectCell=node('td'),actions=node('div',undefined,'row-actions');actions.appendChild(solo);actions.appendChild(inspect);inspectCell.appendChild(actions);row.appendChild(inspectCell);
    checkbox.onchange=()=>{state.solutions[index].enabled=checkbox.checked;refresh();};color.oninput=()=>{state.solutions[index].color=color.value;refresh();};opacity.onchange=()=>{const v=Number(opacity.value);if(Number.isFinite(v))state.solutions[index].opacity=Math.max(0,Math.min(100,v))/100;refresh();};
    rows.push({row,checkbox,color,opacity,warning,inspect,solo});body.appendChild(row);
  });
  function entry(list,label,value){const dt=node('dt',label),dd=node('dd',String(value));list.appendChild(dt);list.appendChild(dd);}
  function updateInspector(){
    const solution=data.solutions[selected],s=solution.settings,provenance=solution.provenance;
    rows.forEach((r,i)=>{r.row.classList.toggle('inspected',i===selected);r.inspect.setAttribute('aria-pressed',String(i===selected));});
    byId('inspector-title').textContent=solution.label;
    byId('inspector-objective').textContent=solution.objective;
    const list=byId('inspector-facts');list.replaceChildren();
    entry(list,'Saved method',solution.id);
    entry(list,'Recording frames',`${data.frame_ids[0]}â€“${data.frame_ids.at(-1)} (${data.frame_ids.length} frames)`);
    entry(list,'Source Parquet',provenance.recording?.path??data.recording.path);
    entry(list,'Recording SHA-256',provenance.recording?.sha256??data.recording.sha256);
    entry(list,'Native binary SHA-256',provenance.native_sha256??'Not recorded');
    if(solution.processing){const p=solution.processing;entry(list,'Processing',`${p.active_frames} active frames, up to ${p.boundary_frames} fixed history frames; relative cost tolerance ${p.function_tolerance}; maximum ${p.max_iterations} iterations per solve.`);entry(list,'Time accounting',`Window Ceres total ${p.window_solve_seconds.toFixed(3)} s; final pass Ceres ${p.final_pass_seconds.toFixed(3)} s; processing wall ${p.wall_seconds.toFixed(3)} s.`);entry(list,'Objective accounting',p.evaluation_scope);entry(list,'Boundary policy',p.boundary);entry(list,'Lookahead',p.latency);}
    if(solution.benchmark_timing)for(const [key,value] of Object.entries(solution.benchmark_timing))entry(list,key,typeof value==='number'?value.toFixed(3):value);
    if(s.shared_axial_length?.bound_fractions)entry(list,'Shared total bounds',`${s.shared_axial_length.minimum_total_mm.toFixed(2)} to ${s.shared_axial_length.maximum_total_mm.toFixed(2)} mm; experimental bounds, not anatomical limits.`);
    if(s.relative_twist_priors?.length)entry(list,'Relative twist preference',s.relative_twist_priors.map(p=>`${p.parent} -> ${p.child}: scale ${p.scale}`).join('; '));
    if(s.sc_anterior_prior)entry(list,'SC anterior preference',`One-sided residual scale ${s.sc_anterior_prior.scale_mm} mm. ${s.sc_anterior_prior.source}`);
    if(s.landmark_position_priors)entry(list,'Axial landmark positions',`${s.landmark_position_priors.landmarks.join(', ')}; XYZ residual scale ${s.landmark_position_priors.scale_mm} mm. ${s.landmark_position_priors.source}`);
    if(s.landmark_huber_scale_mm)entry(list,'Keypoint robust loss',`Huber transition at ${s.landmark_huber_scale_mm} mm Euclidean error per mapped target. Time weights preserve this physical threshold. Targets remain visible and unchanged.`);
    entry(list,'Equal spine lengths',s.length_equality_prior?.enabled?`Sacrolumbar minus thoracic; residual scale ${s.length_equality_prior.scale_mm} mm. Total length remains free.`:'Disabled');
    if(s.segment_axis_prior)entry(list,'SC / shoulder axis preference',`Transverse chord residual scale ${s.segment_axis_prior.scale}; ${s.segment_axis_prior.supported_frames} supported frames. ${s.segment_axis_prior.source}`);
    if(s.shared_axial_length)entry(list,'Shared spine length',`One total length parameter per frame. Exact proportions ${s.shared_axial_length.ratios.join(' : ')}. Existing segment rest-length residuals retained.`);
    if(s.length_proportion_prior?.enabled){const p=s.length_proportion_prior;entry(list,'Spine proportions',`${p.segment_names.join(' : ')} = ${p.ratios.join(' : ')}. Residual scale ${p.scale_mm} mm. Total length free. ${p.source}`);}
    entry(list,'Spine length policy',s.shared_axial_length?.bound_fractions?'Shared total has explicit bounds; length smoothing remains disabled':s.free_axial_lengths?'Nonnegative lengths; no upper bound; length smoothing disabled':`Shortening residual scale: ${s.length_prior_fraction*100}%; lengthening residual scale: ${(s.lengthening_prior_fraction??s.length_prior_fraction)*100}% of reference length`);
    entry(list,'Rest-length preference',(!s.free_axial_lengths||s.free_length_rest_prior)?`Scale ${s.length_prior_fraction*100}% of each saved reference length; symmetric in these comparisons. This is a soft penalty, not a bound.`:'Disabled');
    entry(list,'Chest-center line preference',s.chest_line_prior?.enabled?`Distance scale ${s.chest_line_prior.distance_scale_mm} mm; extra anterior scale ${s.chest_line_prior.anterior_scale_mm} mm`:'Disabled');
    entry(list,'Position residual scale',s.position_scale_mm+' mm');
    entry(list,'Rest-pose residual scale',s.rest_pose_scale_radians+' rad');
    if(s.shoulder_geometry){const g=s.shoulder_geometry;entry(list,'SC lowering',`${(g.lowering_fraction*100).toFixed(0)}% of reference thoracic extent (${(g.lowering_fraction*g.thoracic_reference_length_mm).toFixed(2)} mm)`);entry(list,'SC forward offset',`${(g.forward_factor*100).toFixed(0)}% of original reference offset`);}
    if(s.relaxed_linkage_children?.length){entry(list,'Linkage displacement residual scale',s.linkage_scale_mm+' mm (not a hard bound)');entry(list,'Linkage acceleration residual scale',s.linkage_acceleration_scale_mm_s2+' mm/sÂ²');}
    entry(list,'Residual costs','Ceres half-squared weighted residual sums; objectives with different scales are not directly comparable accuracy scores.');
    for(const [name,value] of Object.entries(s.costs_by_family||{}))entry(list,'Residual cost: '+name,(Array.isArray(value)?value.reduce((a,b)=>a+b,0):value).toFixed(4));
    const metrics=byId('inspector-metrics');metrics.replaceChildren();for(const [k,v] of Object.entries(solution.summary))entry(metrics,k,typeof v==='number'?v.toFixed(3):v);
    byId('inspector-report').textContent=solution.report+(solution.processing?'\n'+solution.processing.final_pass_report:'');
    byId('inspector-settings').textContent=JSON.stringify(s,null,2);
    updateWarning();
  }
  function updateWarning(){
    const diagnostics=data.solutions[selected].frames[reviewIndex].diagnostics;
    if(byId('inspector-axis-angles'))byId('inspector-axis-angles').textContent=Object.entries(diagnostics).filter(([k])=>k.includes('axis')&&k.includes('(degrees)')||k==='Shoulder versus SC axial angle (degrees)'||k.includes('Relative axial twist')||k.includes('anterior to shoulder midpoint')).map(([k,v])=>`${k}: ${Number.isFinite(v)?v.toFixed(1):'undefined'}`).join(' | ');
    const a=diagnostics['sacrolumbar length (mm)'],b=diagnostics['thoracic length (mm)'];
    byId('inspector-warning').textContent=[diagnostics['Spine fit warning'],diagnostics['Pose support warning']].filter(Boolean).join(' ')||'No saved spine warning for this frame.';
    const processing=data.solutions[selected].processing;
    byId('processing-report').hidden=!processing;byId('processing-band').hidden=!processing;
    byId('inspector-processing').textContent='';
    if(processing){
      const w=processing.windows[diagnostics['Finalizing window']];
      if(w){
        byId('inspector-processing').textContent=`Recorded solve window ${w.index+1}/${processing.windows.length}: active frames ${data.frame_ids[w.active_start]} to ${data.frame_ids[w.active_end]}; ${w.active_start-w.fixed_start} fixed history frames.\n${w.iterations} Ceres iterations; ${w.seconds.toFixed(3)} s; ${w.converged?'converged':'stopping limit, inspect report'}.\n${processing.refined?'Displayed pose includes subsequent full-recording refinement.':'Displayed pose is the finalized window result.'} Window costs overlap; the global score evaluates each residual once.`;
        byId('window-report').textContent=w.full_report;
        const a=100*w.fixed_start/data.times.length,b=100*w.active_start/data.times.length,c=100*(w.active_end+1)/data.times.length;
        byId('processing-band').style.background=`linear-gradient(to right,#374859 0% ${a}%,#e4b66a ${a}% ${b}%,#73d6e6 ${b}% ${c}%,#243347 ${c}% 100%)`;
      }
    }
    const neckHeight=diagnostics['Neck center above shoulder midpoint (mm)'],scHeight=diagnostics['SC midpoint above shoulder midpoint (mm)'];
    byId('inspector-shoulders').textContent=Number.isFinite(neckHeight)&&Number.isFinite(scHeight)?`Along the hip-to-shoulder axis: neck center ${neckHeight.toFixed(1)} mm above the shoulder midpoint; SC midpoint ${scHeight.toFixed(1)} mm above it. Positive means toward the head; this is a geometric comparison, not anatomical ground truth.`:'';
    const c=diagnostics['cervical_spine length (mm)'];
    byId('inspector-lengths').textContent=Number.isFinite(a)&&Number.isFinite(b)?`Frame ${data.frame_ids[reviewIndex]}: sacrolumbar ${a.toFixed(1)} mm; thoracic ${b.toFixed(1)} mm; ${Number.isFinite(c)?'cervical '+c.toFixed(1)+' mm; ':''}difference ${(a-b).toFixed(1)} mm.`:'';
    const prior=data.solutions[selected].settings.length_proportion_prior,total=a+b+c;
    if(Number.isFinite(total)){
      byId('inspector-lengths').textContent+=` Total spine length: ${total.toFixed(1)} mm.`;
      const bounds=data.solutions[selected].settings.shared_axial_length;
      if(bounds?.bound_fractions)byId('inspector-lengths').textContent+=` Allowed total: ${bounds.minimum_total_mm.toFixed(1)} to ${bounds.maximum_total_mm.toFixed(1)} mm.`;
    }
    if((prior?.enabled||data.solutions[selected].settings.shared_axial_length)&&Number.isFinite(total)&&total>0)byId('inspector-lengths').textContent+=` Actual shares: ${[a,b,c].map(v=>(100*v/total).toFixed(1)+'%').join(' / ')}; target shares: ${prior.fractions.map(v=>(100*v).toFixed(1)+'%').join(' / ')} (sacrolumbar / thoracic / cervical).`;
  }
  function update(){
    rows.forEach((r,i)=>{const setting=state.solutions[i];r.checkbox.checked=setting.enabled;r.color.value=setting.color;r.opacity.value=Math.round(setting.opacity*100);r.row.style.setProperty('--color',setting.color);r.row.classList.toggle('enabled',setting.enabled);r.warning.textContent=data.solutions[i].frames[reviewIndex].diagnostics['Spine fit warning']?' !':'';});updateWarning();
    byId('inventory-summary').textContent=`${data.solutions.length} saved fits Â· frames ${data.frame_ids[0]}â€“${data.frame_ids.at(-1)} Â· ${state.solutions.filter(s=>s.enabled&&s.opacity>0).length} visible`;
  }
  updateInspector();return {update,rows};
}
