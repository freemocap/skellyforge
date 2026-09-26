/* UI for already-solved, frame-aligned recordings. */
const reviewData=RECORDING_COMPARISON;
const byId=id=>document.getElementById(id);
const reviewState={region:'all',grid:true,keypoints:true,landmarks:true,segments:true,linkages:true,savedLandmarks:false,savedSegments:false,axes:false,
  solutions:reviewData.solutions.map(s=>({enabled:(reviewData.solutions.some(v=>v.id==='equal_spine_lengths')?['lower_sc_relaxed','equal_spine_lengths']:['lower_sc','lower_sc_relaxed']).includes(s.id),opacity:s.id==='equal_spine_lengths'?.9:.65,color:s.color}))};
if(!reviewState.solutions.some(s=>s.enabled))reviewState.solutions.at(-1).enabled=true;
const initialChoices=reviewState.solutions.map(s=>({...s}));
let reviewIndex=0,reviewPlaying=false,playOrigin=0,playTime=0;
const viewer=createComparisonScene(reviewData,byId('scene'),byId('tooltip'));
function node(tag,text,className){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(className)n.className=className;return n;}
const fitTable=createComparisonTable(reviewData,reviewState,refresh);
byId('recording-name').textContent=reviewData.recording.path.split(/[\\/]/).at(-2);
byId('timeline').max=reviewData.times.length-1;
byId('frame-number').min=reviewData.frame_ids[0];byId('frame-number').max=reviewData.frame_ids.at(-1);
byId('start-label').textContent='Frame '+reviewData.frame_ids[0];byId('end-label').textContent='Frame '+reviewData.frame_ids.at(-1);
const imageCache=new Map();let desiredVideo='',shownVideo='';
function preparedImage(url){
  if(imageCache.has(url))return imageCache.get(url);
  const image=new Image();const promise=new Promise((resolve,reject)=>{image.onload=()=>{(image.decode?image.decode():Promise.resolve()).then(()=>resolve(image),reject);};image.onerror=()=>reject(new Error('Preview unavailable'));});
  imageCache.set(url,promise);image.src=url;promise.catch(()=>{});
  while(imageCache.size>64)imageCache.delete(imageCache.keys().next().value);
  return promise;
}
function refreshVideo(){
  const enabled=byId('show-video').checked,view=reviewData.videos[Number(byId('camera').value)];
  byId('video-box').hidden=!enabled;byId('camera').disabled=!enabled;
  if(!view||!enabled){desiredVideo='';byId('video-status').textContent=view?'Video hidden':'No annotated previews available';return;}
  const number=reviewData.frame_ids[reviewIndex],url=n=>view.base+String(n).padStart(6,'0')+'.jpg?v='+view.sha256.slice(0,12),key=url(number);
  if(desiredVideo===key)return;desiredVideo=key;
  byId('video-status').textContent=`Loading frame ${number}${shownVideo?'; previous image retained':''}…`;
  preparedImage(key).then(image=>{
    if(desiredVideo!==key)return;
    const displayed=byId('video-image');displayed.src=image.src;displayed.hidden=false;displayed.dataset.frame=String(number);shownVideo=key;byId('video-placeholder').hidden=true;
    byId('video-status').textContent=`Frame ${number} · ${reviewData.times[reviewData.frame_ids.indexOf(number)].toFixed(3)} s · annotated`;
  },()=>{if(desiredVideo===key)byId('video-status').textContent=`Frame ${number} unavailable${shownVideo?'; previous image retained':''}`;});
  for(let i=reviewIndex+1;i<Math.min(reviewIndex+5,reviewData.frame_ids.length);i++)preparedImage(url(reviewData.frame_ids[i]));
}
reviewData.videos.forEach((view,i)=>{const option=node('option',view.label);option.value=String(i);byId('camera').appendChild(option);});byId('camera').value='0';
byId('camera').onchange=refreshVideo;byId('show-video').onchange=refreshVideo;
function refresh(){
  viewer.draw(reviewIndex,reviewState);
  byId('timeline').value=reviewIndex;byId('frame-number').value=reviewData.frame_ids[reviewIndex];
  byId('frame-label').textContent=`Frame ${reviewData.frame_ids[reviewIndex]} · ${reviewData.times[reviewIndex].toFixed(3)} s`;
  let count=0;byId('overlay-legend').replaceChildren();
  fitTable.update();
  reviewState.solutions.forEach((setting,i)=>{if(setting.enabled&&setting.opacity>0){count++;const chip=node('span',reviewData.solutions[i].label,'legend-chip');chip.style.setProperty('--color',setting.color);byId('overlay-legend').appendChild(chip);}});
  byId('active-count').textContent=count?`${count} solution${count===1?'':'s'} overlaid`:'No solutions enabled';refreshVideo();
}
function pauseReview(){reviewPlaying=false;byId('play').textContent='Play';byId('play').setAttribute('aria-pressed','false');}
function seekReview(index){pauseReview();reviewIndex=Math.max(0,Math.min(reviewData.times.length-1,index));refresh();}
function toggleReview(){if(reviewPlaying){pauseReview();return;}if(reviewIndex===reviewData.times.length-1)reviewIndex=0;playTime=reviewData.times[reviewIndex];playOrigin=performance.now();reviewPlaying=true;byId('play').textContent='Pause';byId('play').setAttribute('aria-pressed','true');refresh();}
byId('play').onclick=toggleReview;byId('timeline').oninput=()=>seekReview(Number(byId('timeline').value));
byId('frame-number').onchange=()=>{const wanted=Number(byId('frame-number').value);if(!Number.isFinite(wanted)){refresh();return;}seekReview(reviewData.frame_ids.reduce((best,n,i)=>Math.abs(n-wanted)<Math.abs(reviewData.frame_ids[best]-wanted)?i:best,0));};
byId('first').onclick=()=>seekReview(0);byId('last').onclick=()=>seekReview(reviewData.times.length-1);byId('previous').onclick=()=>seekReview(reviewIndex-1);byId('next').onclick=()=>seekReview(reviewIndex+1);
byId('speed').onchange=()=>{playTime=reviewData.times[reviewIndex];playOrigin=performance.now();};
byId('hide-all').onclick=()=>{reviewState.solutions.forEach(s=>s.enabled=false);refresh();};byId('reset-solutions').onclick=()=>{reviewState.solutions=initialChoices.map(s=>({...s}));refresh();};
byId('reset-camera').onclick=viewer.reset;byId('region').onchange=()=>{reviewState.region=byId('region').value;refresh();};
for(const [id,key] of Object.entries({'keypoints':'keypoints','landmarks':'landmarks','segments':'segments','linkages':'linkages','saved-landmarks':'savedLandmarks','saved-segments':'savedSegments','axes':'axes','grid':'grid'}))byId('show-'+id).onchange=()=>{reviewState[key]=byId('show-'+id).checked;refresh();};
document.addEventListener('keydown',event=>{if(/INPUT|SELECT|BUTTON|TEXTAREA/.test(event.target.tagName))return;if(event.code==='Space'){event.preventDefault();toggleReview();}else if(event.key==='ArrowLeft'){event.preventDefault();seekReview(reviewIndex-1);}else if(event.key==='ArrowRight'){event.preventDefault();seekReview(reviewIndex+1);}});
function setupDivider(id,property,value,min,max,fromPointer,keyDirection){
  const element=byId(id),apply=v=>{value=Math.max(min,Math.min(max(),v));document.documentElement.style.setProperty(property,value+'px');element.setAttribute('aria-valuenow',Math.round(value));element.setAttribute('aria-valuemin',min);element.setAttribute('aria-valuemax',Math.round(max()));};
  element.addEventListener('pointerdown',e=>{if(e.button!==0)return;element.setPointerCapture(e.pointerId);element.classList.add('dragging');document.body.classList.add('resizing');e.preventDefault();});
  element.addEventListener('pointermove',e=>{if(element.hasPointerCapture(e.pointerId))apply(fromPointer(e));});
  const release=e=>{if(element.hasPointerCapture(e.pointerId))element.releasePointerCapture(e.pointerId);element.classList.remove('dragging');document.body.classList.remove('resizing');};
  element.addEventListener('pointerup',release);element.addEventListener('pointercancel',release);element.addEventListener('lostpointercapture',release);
  element.addEventListener('keydown',e=>{if(keyDirection[e.key]){e.preventDefault();apply(value+keyDirection[e.key]*20);}});window.addEventListener('resize',()=>apply(value));apply(value);return ()=>apply(value);
}
setupDivider('sidebar-divider','--sidebar',740,330,()=>Math.max(330,innerWidth-400),e=>innerWidth-e.clientX,{ArrowLeft:1,ArrowRight:-1});
const resizeVideo=setupDivider('video-divider','--video-width',240,120,()=>Math.max(120,byId('media-workspace').clientWidth-207),e=>e.clientX-byId('media-workspace').getBoundingClientRect().left,{ArrowLeft:-1,ArrowRight:1});
new ResizeObserver(resizeVideo).observe(byId('media-workspace'));
function advanceReview(now){
  if(!reviewPlaying)return;
  const first=reviewData.times[0],last=reviewData.times.at(-1),dt=reviewData.times.length>1?last-reviewData.times.at(-2):1,duration=last-first+dt;
  let time=playTime+(now-playOrigin)/1000*Number(byId('speed').value);
  if(time>=first+duration){if(byId('loop').checked)time=first+(time-first)%duration;else{reviewIndex=reviewData.times.length-1;pauseReview();refresh();return;}}
  let index=0;while(index+1<reviewData.times.length&&reviewData.times[index+1]<=time)index++;
  if(index!==reviewIndex){reviewIndex=index;refresh();}
}
function animateReview(now){advanceReview(now);viewer.render();requestAnimationFrame(animateReview);}
refresh();requestAnimationFrame(animateReview);
