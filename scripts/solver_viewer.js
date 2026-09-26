// Experiment controller. Each experiment supplies runs, controls and segment definitions.
let experiment,selectedSequence,selectedMode,current;
let frameIndex=0,playing=false,playStart=0;
const savedChoices=new Map();
function options(select,values){select.replaceChildren(...values.map(v=>{const option=document.createElement('option');option.value=String(v.id);option.textContent=v.label;return option;}));}
function stop(){playing=false;el('play').textContent='Play';}
function chooseExperiment(){
  stop();frameIndex=0;experiment=EXPERIMENTS.find(e=>e.id===el('experiment').value);
  el('scene-title').textContent=experiment.label;el('experiment-description').textContent=experiment.description;
  const saved=savedChoices.get(experiment.id)||{};
  const regions=[...new Set(experiment.bodies.map(b=>b.region).filter(Boolean))];
  options(el('body-region'),[{id:'all',label:'Whole body'},...regions.map(r=>({id:r,label:r}))]);
  el('body-region').value=regions.includes('Trunk')?'Trunk':'all';
  chooseRegion();
  el('segment-colors').value=regions.length?'model':'segments';
  const container=el('experiment-controls');container.replaceChildren();
  for(const control of experiment.controls){
    const label=document.createElement('label');label.textContent=control.label;
    const select=document.createElement('select');select.id='parameter-'+control.id;
    options(select,control.values.map(v=>({id:v,label:String(v)})));
    select.value=String(saved[control.id]??(control.id==='noise'?1:control.values[0]));
    select.addEventListener('change',()=>{saved[control.id]=Number(select.value);savedChoices.set(experiment.id,saved);stop();frameIndex=0;selectRun();});label.appendChild(select);container.appendChild(label);
  }
  options(el('mode'),experiment.methods);el('mode').value=saved.method||experiment.default_method||experiment.methods.at(-1).id;
  options(el('plot-body'),experiment.bodies.map((b,i)=>({id:i,label:b.label})));el('plot-body').value='0';
  options(el('video-camera'),(experiment.annotated_views||[]).map((v,i)=>({id:i,label:v.label})));el('video-camera').value='0';
  el('annotated-frame').dataset.key='';
  selectBody();selectRun();reset();
}
function selectBody(){const definition=experiment.bodies[Number(el('plot-body').value)];options(el('plot-point'),definition.landmark_names.map((label,i)=>({id:i,label})));el('plot-point').value='0';}
function chooseRegion(){
  options(el('body-focus'),[{id:'all',label:'All segments in region'},...experiment.bodies.map((b,i)=>({id:i,label:b.label,region:b.region})).filter(b=>el('body-region').value==='all'||b.region===el('body-region').value)]);
  el('body-focus').value='all';
}
function selectRun(){
  selectedSequence=experiment.runs.find(run=>experiment.controls.every(c=>run.parameters[c.id]===Number(el('parameter-'+c.id).value)));
  selectedMode=selectedSequence.methods[el('mode').value];
  el('playback').hidden=selectedMode.frames.length===1;el('frame').max=selectedMode.frames.length-1;
  draw();
}
function displayValues(values){return Object.entries(values||{}).map(([k,v])=>`${k}: ${typeof v==='number'?v.toFixed(3):v}`).join('\n');}
function draw(){
  current=selectedMode.frames[frameIndex];drawScene(current,experiment);
  el('frame').value=frameIndex;el('clock').textContent=`Frame ${frameIndex} / ${selectedMode.frames.length-1} | ${selectedSequence.times[frameIndex].toFixed(2)} s`;
  const recordingFrame=current.diagnostics?.['Recording frame'];
  el('recording-frame-label').hidden=recordingFrame===undefined;
  if(recordingFrame!==undefined){
    el('recording-frame').value=recordingFrame;
    el('recording-frame').min=selectedMode.frames[0].diagnostics['Recording frame'];
    el('recording-frame').max=selectedMode.frames.at(-1).diagnostics['Recording frame'];
    el('clock').textContent=`Recording frame ${recordingFrame} | window frame ${frameIndex} / ${selectedMode.frames.length-1}`;
  }
  const body=current.bodies[Number(el('plot-body').value)];
  const residuals=current.bodies.flatMap(b=>b.residuals).filter(r=>r!==null);
  const targetRMS=Math.sqrt(residuals.reduce((sum,r)=>sum+r*r,0)/residuals.length);
  const available=current.bodies.filter(b=>b.available!==false&&b.truth_rms!==null);
  const truthRMS=Math.sqrt(available.reduce((sum,b)=>sum+b.truth_rms*b.truth_rms*b.fitted.length,0)/available.reduce((sum,b)=>sum+b.fitted.length,0));
  el('status').textContent=`${current.converged?'CONVERGED':'NOT CONVERGED'}\nTarget RMS: ${targetRMS.toFixed(3)} mm\nAvailable fits vs known RMS: ${available.length?truthRMS.toFixed(3)+" mm":"not available (real recording)"}\n${displayValues(current.diagnostics)}\nSolve time: ${(current.seconds*1000).toFixed(2)} ms`;
  el('sequence-status').textContent=displayValues(selectedMode.summary);el('sequence-status').hidden=!Object.keys(selectedMode.summary).length;
  el('observability').textContent=current.observability||'';el('observability').hidden=!current.observability;
  el('settings').textContent=JSON.stringify(selectedMode.settings,null,2)+'\n'+current.report;
  el('poses').textContent=JSON.stringify(experiment.bodies.map((definition,i)=>({segment:definition.label,quaternion:current.bodies[i].quaternion,translation:current.bodies[i].translation})),null,2);
  el('errors').innerHTML=experiment.bodies[Number(el('plot-body').value)].landmark_names.map((name,i)=>`<tr><th>${name}</th><td>${body.residuals[i]!=null?body.residuals[i].toFixed(3):'No measurement residual'}</td></tr>`).join('');
  el('objective-caption').textContent=selectedMode.objective;
  const costs=current.costs;el('cost').hidden=!costs.length;
  if(costs.length){const max=Math.max(...costs,1e-20);const coords=costs.map((c,i)=>`${10+320*i/Math.max(costs.length-1,1)},${80-70*c/max}`).join(' ');el('cost').innerHTML=`<polyline points="${coords}" fill="none" stroke="#ffac54" stroke-width="2"/><text x="12" y="14" fill="white" font-size="10">${costs[0].toPrecision(4)} → ${costs.at(-1).toPrecision(4)}</text>`;}
  drawCeresProblem();
  drawTimeSeries();drawAnnotatedFrame();
}
// Decode offscreen; retain the displayed frame until its replacement is ready.
const annotatedFrames=new Map();
function prepareAnnotatedFrame(url){
  let entry=annotatedFrames.get(url);if(entry)return entry;
  const image=document.createElement('img');entry={image,ready:false,failed:false,callbacks:[]};annotatedFrames.set(url,entry);
  while(annotatedFrames.size>12)annotatedFrames.delete(annotatedFrames.keys().next().value);
  const finish=()=>{entry.ready=true;entry.callbacks.splice(0).forEach(fn=>fn());};
  image.onload=()=>{if(image.decode)image.decode().then(finish,()=>{entry.failed=true;entry.callbacks.splice(0).forEach(fn=>fn());});else finish();};
  image.onerror=()=>{entry.failed=true;entry.callbacks.splice(0).forEach(fn=>fn());};
  image.src=url;return entry;
}
function drawAnnotatedFrame(){
  const views=experiment.annotated_views||[];
  el('video-panel').hidden=!views.length;
  if(!views.length)return;
  const view=views[Number(el('video-camera').value)],number=current.diagnostics['Recording frame'];
  const img=el('annotated-frame'),key=experiment.id+'|'+view.base+'|'+number;
  if(img.dataset.key===key)return;
  img.dataset.key=key;
  const caption=`Recording frame ${number} | ${current.diagnostics['Recording timestamp (s)'].toFixed(3)} s`;
  const urlFor=n=>view.base+String(n).padStart(6,'0')+'.jpg?v='+view.sha256.slice(0,12);
  const entry=prepareAnnotatedFrame(urlFor(number));
  const present=()=>{
    if(img.dataset.key!==key||experiment.annotated_views?.[Number(el('video-camera').value)]!==view)return;
    if(entry.failed){el('video-value').textContent=caption+' | preview unavailable; previous image retained';return;}
    img.src=entry.image.src;img.hidden=false;img.dataset.displayed=String(number);
    el('video-value').textContent=caption+' | annotated source, matched frame number';
  };
  if(entry.ready||entry.failed)present();else{
    el('video-value').textContent=caption+` | loading; displayed frame ${img.dataset.displayed||'none'} retained`;
    entry.callbacks.push(present);
  }
  for(let n=number+1;n<Math.min(number+4,view.count);n++)prepareAnnotatedFrame(urlFor(n));
}
el('video-camera').addEventListener('change',drawAnnotatedFrame);
options(el('experiment'),EXPERIMENTS);el('experiment').value=EXPERIMENTS.some(e=>e.id==='recording_shoulder_linkages')?'recording_shoulder_linkages':EXPERIMENTS.some(e=>e.id==='recording_shoulders')?'recording_shoulders':EXPERIMENTS.some(e=>e.id==='recording_body')?'recording_body':EXPERIMENTS.some(e=>e.id==='recording_torso')?'recording_torso':'torso';
el('body-region').addEventListener('change',()=>{chooseRegion();ceresMap.autoFit=true;draw();});
el('body-focus').addEventListener('change',()=>{ceresMap.autoFit=true;draw();});
el('segment-colors').addEventListener('change',draw);
el('experiment').addEventListener('change',chooseExperiment);
el('mode').addEventListener('change',()=>{stop();const saved=savedChoices.get(experiment.id)||{};saved.method=el('mode').value;savedChoices.set(experiment.id,saved);selectRun();});
el('plot-body').addEventListener('change',()=>{selectBody();draw();});
for(const k of ['truth','initial','fitted','observed','residuals','axes','contextSegments','contextLandmarks','contextKeypoints','chestCenterline'])el(k).addEventListener('change',draw);
el('frame').oninput=()=>{stop();frameIndex=Number(el('frame').value);draw();};
el('recording-frame').addEventListener('change',()=>{
  const index=selectedMode.frames.findIndex(f=>f.diagnostics?.['Recording frame']===Number(el('recording-frame').value));
  if(index>=0){stop();frameIndex=index;}draw();
});
el('previous').onclick=()=>{stop();frameIndex=Math.max(0,frameIndex-1);draw();};
el('next').onclick=()=>{stop();frameIndex=Math.min(selectedMode.frames.length-1,frameIndex+1);draw();};
el('play').onclick=()=>{if(playing){stop();return;}if(frameIndex===selectedMode.frames.length-1)frameIndex=0;playing=true;playStart=performance.now()-selectedSequence.times[frameIndex]*1000;el('play').textContent='Pause';};
el('reset').onclick=reset;
el('download').onclick=()=>{const payload={schema_version:1,experiment:{id:experiment.id,description:experiment.description,bodies:experiment.bodies,metadata:experiment.metadata},...selectedSequence,selected_method:el('mode').value};const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`forge-${experiment.id}-run.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
setupScene();setupSidebar();setupTimeSeries();reset();chooseExperiment();
function animate(){requestAnimationFrame(animate);if(playing){const time=(performance.now()-playStart)/1000;let next=frameIndex;while(next+1<selectedSequence.times.length&&selectedSequence.times[next+1]<=time)next++;if(next!==frameIndex){frameIndex=next;draw();}if(next===selectedMode.frames.length-1)stop();}controls.update();updateAxisLabels();renderer.render(scene,camera);}animate();
