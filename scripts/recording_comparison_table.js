/* Audit table: describe saved solver settings, not inferred anatomy. */
function describeComparisonFit(solution){
  const s=solution.settings,g=s.shoulder_geometry;
  const free=s.free_axial_lengths===true,short=s.length_prior_fraction,long=s.lengthening_prior_fraction??short;
  return {
    spine:free?'Free lengths':'Length priors',
    spineDetail:free?(s.chest_line_prior?.enabled?'Centerline preference':'No centerline preference'):`Prior scales ${short*100}% / ${long*100}%`,
    offsets:!g?'Original':g.forward_factor<1?'Lower + closer':'Lowered',
    shoulders:s.relaxed_linkage_children?.length?'Relaxed':'Exact',
    shoulderDetail:s.relaxed_linkage_children?.length?`${s.linkage_scale_mm} mm residual scale`:'Coincident attachments',
    rms:solution.summary['Observed landmark RMS (mm)'],
  };
}
function createComparisonTable(data,state,refresh){
  const rows=[];let selected=data.solutions.findIndex(s=>s.id==='lower_sc_relaxed');if(selected<0)selected=0;
  const table=node('table');table.className='fit-table';const caption=node('caption','Saved fits — identical recording frames, different solver settings');caption.className='sr-only';table.appendChild(caption);
  const head=node('thead'),headRow=node('tr');
  for(const text of ['Display','Spine','SC reference','Shoulder linkage','Solve','Target RMS','Actions']){const th=node('th',text);th.setAttribute('scope','col');headRow.appendChild(th);}
  head.appendChild(headRow);table.appendChild(head);const body=node('tbody');table.appendChild(body);byId('solutions').appendChild(table);
  function cell(row,primary,secondary){const td=node('td');td.appendChild(node('div',primary));if(secondary)td.appendChild(node('div',secondary,'cell-detail'));row.appendChild(td);return td;}
  data.solutions.forEach((solution,index)=>{
    const description=describeComparisonFit(solution),row=node('tr');row.setAttribute('aria-label',solution.label);row.style.setProperty('--color',solution.color);
    const display=node('td'),displayControls=node('div',undefined,'row-display');
    const checkbox=node('input');checkbox.type='checkbox';checkbox.setAttribute('aria-label','Overlay '+solution.label);displayControls.appendChild(checkbox);
    const color=node('input');color.type='color';color.setAttribute('aria-label',solution.label+' color');displayControls.appendChild(color);
    const opacity=node('input');opacity.type='number';opacity.min=0;opacity.max=100;opacity.step=5;opacity.setAttribute('aria-label',solution.label+' opacity percent');opacity.title='Opacity (%)';displayControls.appendChild(opacity);display.appendChild(displayControls);
    const solo=node('button','Solo');solo.setAttribute('aria-label','Show only '+solution.label);solo.onclick=()=>{state.solutions.forEach((s,i)=>s.enabled=i===index);refresh();};row.appendChild(display);
    const spineCell=cell(row,description.spine,description.spineDetail);spineCell.title='Prior scales are shortening / lengthening residual scales, relative to reference length; not hard bounds.';cell(row,description.offsets);cell(row,description.shoulders,description.shoulderDetail);
    const result=cell(row,solution.converged?'Converged':'Not converged',solution.seconds.toFixed(1)+' s');result.className=solution.converged?'result-ok':'result-warning';const warning=node('div','','frame-warning');result.appendChild(warning);
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
    entry(list,'Recording frames',`${data.frame_ids[0]}–${data.frame_ids.at(-1)} (${data.frame_ids.length} frames)`);
    entry(list,'Source Parquet',provenance.recording?.path??data.recording.path);
    entry(list,'Recording SHA-256',provenance.recording?.sha256??data.recording.sha256);
    entry(list,'Native binary SHA-256',provenance.native_sha256??'Not recorded');
    entry(list,'Spine length policy',s.free_axial_lengths?'Nonnegative lengths; no length prior or upper bound':`Shortening residual scale: ${s.length_prior_fraction*100}%; lengthening residual scale: ${(s.lengthening_prior_fraction??s.length_prior_fraction)*100}% of reference length`);
    entry(list,'Chest-center line preference',s.chest_line_prior?.enabled?`Distance scale ${s.chest_line_prior.distance_scale_mm} mm; extra anterior scale ${s.chest_line_prior.anterior_scale_mm} mm`:'Disabled');
    entry(list,'Position residual scale',s.position_scale_mm+' mm');
    entry(list,'Rest-pose residual scale',s.rest_pose_scale_radians+' rad');
    if(s.shoulder_geometry){const g=s.shoulder_geometry;entry(list,'SC lowering',`${(g.lowering_fraction*100).toFixed(0)}% of reference thoracic extent (${(g.lowering_fraction*g.thoracic_reference_length_mm).toFixed(2)} mm)`);entry(list,'SC forward offset',`${(g.forward_factor*100).toFixed(0)}% of original reference offset`);}
    if(s.relaxed_linkage_children?.length){entry(list,'Linkage displacement residual scale',s.linkage_scale_mm+' mm (not a hard bound)');entry(list,'Linkage acceleration residual scale',s.linkage_acceleration_scale_mm_s2+' mm/s²');}
    const metrics=byId('inspector-metrics');metrics.replaceChildren();for(const [k,v] of Object.entries(solution.summary))entry(metrics,k,typeof v==='number'?v.toFixed(3):v);
    byId('inspector-report').textContent=solution.report;
    byId('inspector-settings').textContent=JSON.stringify(s,null,2);
    updateWarning();
  }
  function updateWarning(){byId('inspector-warning').textContent=data.solutions[selected].frames[reviewIndex].diagnostics['Spine fit warning']||'No saved spine warning for this frame.';}
  function update(){
    rows.forEach((r,i)=>{const setting=state.solutions[i];r.checkbox.checked=setting.enabled;r.color.value=setting.color;r.opacity.value=Math.round(setting.opacity*100);r.row.style.setProperty('--color',setting.color);r.row.classList.toggle('enabled',setting.enabled);r.warning.textContent=data.solutions[i].frames[reviewIndex].diagnostics['Spine fit warning']?'Zero spine length':'';});updateWarning();
    byId('inventory-summary').textContent=`${data.solutions.length} saved fits · frames ${data.frame_ids[0]}–${data.frame_ids.at(-1)} · ${state.solutions.filter(s=>s.enabled&&s.opacity>0).length} visible`;
  }
  updateInspector();return {update,rows};
}
