// Presentation only. Sign alignment changes quaternion display, never solver data.
function continuousQuaternions(values, firstReference){
  let previous=firstReference;
  return values.map(q=>{if(!q)return null;previous??=q;const sign=q.reduce((s,v,i)=>s+v*previous[i],0)<0?-1:1;const aligned=q.map(v=>v*sign);previous=aligned;return aligned;});
}
const plotColors={x:'#ff6464',y:'#69d488',z:'#669fff',w:'#b8c2d1',d:'#f2d17b',l:'#78d6a3',t:'#91a9ff'};
const plotConfig={scrollZoom:true,displayModeBar:true,displaylogo:false,responsive:false,
  modeBarButtonsToRemove:['select2d','lasso2d'],toImageButtonOptions:{format:'svg',filename:'forge-trajectory'}};
function drawTimeSeries(){
  const frames=selectedMode.frames.map(f=>f.bodies[Number(el("plot-body").value)]);
  const times=selectedSequence?selectedSequence.times:[0];
  const pointIndex=Number(el('plot-point').value);
  const enabled=['x','y','z'].filter(k=>el('plot-'+k).checked);
  const trueQ=continuousQuaternions(frames.map(f=>f.reference_quaternion),frames[0].reference_quaternion);
  const fittedQ=continuousQuaternions(frames.map(f=>f.quaternion),trueQ[0]);
  const definitions=[
    {id:'translation',keys:enabled,fit:frames.map(f=>f.translation),known:frames.map(f=>f.reference_translation),unit:'mm'},
    {id:'rotation',keys:[...(el('plot-w').checked?['w']:[]),...enabled],fit:fittedQ,known:trueQ,unit:'quaternion component'},
    {id:'point',keys:enabled,fit:frames.map(f=>f.fitted[pointIndex]),known:frames.map(f=>f.truth?.[pointIndex]??null),observed:frames.map(f=>f.observed[pointIndex]||null),unit:'mm'}];
  const hasLine=selectedMode.frames.some(f=>f.chest_line);
  el('chest-line-panel').hidden=!hasLine;
  if(hasLine)definitions.push({id:'chest-line',keys:['x','y'],labels:{x:'lateral',y:'anterior'},
    fit:selectedMode.frames.map(f=>f.chest_line?[f.chest_line.lateral_mm,f.chest_line.anterior_mm]:null),
    known:selectedMode.frames.map(f=>f.chest_line?[0,0]:null),knownLabel:'centerline',unit:'mm'});
  el('length-panel').hidden=!selectedSequence.length_series;
  if(selectedSequence.length_series)definitions.push({id:'length',keys:['l','t'],fit:selectedMode.frames.map(f=>f.lengths),known:selectedMode.frames.map(f=>f.reference_lengths),knownLabel:selectedSequence.length_reference_label||'reference length',unit:'mm'});
  el('displacement-panel').hidden=!selectedSequence.displacement_series;
  if(selectedSequence.displacement_series)definitions.push({id:'displacement',keys:['d'],fit:selectedMode.frames.map(f=>[f.displacement]),known:selectedMode.frames.map(f=>[f.reference_displacement]),unit:'mm'});
  const key=[experiment.id,JSON.stringify(selectedSequence.parameters),el('mode').value,el('plot-body').value,el('plot-point').value].join('|')+['x','y','z','w'].map(k=>el('plot-'+k).checked).join();
  const index=selectedMode?frameIndex:0;
  for(const d of definitions){
    const component=k=>d.id==='length'?['l','t'].indexOf(k):d.id==='displacement'?0:(d.id==='rotation'?['w','x','y','z']:['x','y','z']).indexOf(k);
    const graph=el(d.id+'-series'),t=times[index];
    const cursor={type:'line',xref:'x',yref:'paper',x0:t,x1:t,y0:0,y1:1,line:{color:'#fff9',width:1}};
    if(graph.dataset.plotKey!==key){
      graph.dataset.plotKey=key;
      const traces=[];
      for(const k of d.keys){
        const c=component(k);
        for(const [rows,label,dash] of [[d.fit,'fit','solid'],[d.known,d.knownLabel||'known','dash']].filter(([rows])=>rows.some(row=>row!==null)))
          traces.push({type:'scatter',mode:frames.length===1?'markers':'lines',x:times,y:rows.map(row=>row?row[c]:null),connectgaps:false,
            name:(d.labels?.[k]||k.toUpperCase())+' '+label,uid:k+label,line:{color:plotColors[k],width:dash==='solid'?2:1.5,dash},
            hovertemplate:'%{x:.3f} s<br>%{y:.4f} '+d.unit+'<extra>'+(d.labels?.[k]||k.toUpperCase())+' '+label+'</extra>'});
        if(d.observed)traces.push({type:'scatter',mode:'markers',x:times,y:d.observed.map(row=>row?row[c]:null),
          name:(d.labels?.[k]||k.toUpperCase())+' observed',uid:k+'observed',marker:{color:plotColors[k],size:5,opacity:.55},
          hovertemplate:'%{x:.3f} s<br>%{y:.4f} mm<extra>'+(d.labels?.[k]||k.toUpperCase())+' observed</extra>'});
      }
      const layout={autosize:true,paper_bgcolor:'#192638',plot_bgcolor:'#101c2c',font:{color:'#cbd9ea',size:11},
        margin:{l:55,r:12,t:12,b:42},dragmode:'zoom',hovermode:'closest',
        uirevision:experiment.id+'|'+el('plot-body').value,showlegend:false,
        xaxis:{title:{text:'Time (s)'},gridcolor:'#304156',zerolinecolor:'#526780',automargin:true},
        yaxis:{title:{text:d.unit},gridcolor:'#304156',zerolinecolor:'#526780',automargin:true},shapes:[cursor,...(selectedSequence.gap_times?[{type:"rect",xref:"x",yref:"paper",x0:selectedSequence.gap_times[0],x1:selectedSequence.gap_times[1],y0:0,y1:1,fillcolor:"#e3ac4530",line:{width:0},layer:"below"}]:[])]};
      const first=!graph._fullLayout;
      Plotly.react(graph,traces,layout,plotConfig).then(()=>{
        if(first&&!graph._frameClickBound){graph._frameClickBound=true;graph.on('plotly_click',event=>{
          if(!selectedSequence||!event.points?.length)return;stop();const target=event.points[0].x;
          frameIndex=selectedSequence.times.reduce((best,t,i)=>Math.abs(t-target)<Math.abs(selectedSequence.times[best]-target)?i:best,0);draw();
        });}
      });
    }else if(graph._fullLayout){Plotly.relayout(graph,{'shapes[0].x0':t,'shapes[0].x1':t});}
    el(d.id+'-value').textContent=`t=${t.toFixed(2)}s${d.id==='point'?' | landmark '+pointIndex:''}\n`+
      d.keys.map(k=>`${(d.labels?.[k]||k.toUpperCase())} ${d.fit[index]?d.fit[index][component(k)].toFixed(3):"no fitted pose"} ${d.known[index]?`(${d.knownLabel||"known"} ${d.known[index][component(k)].toFixed(3)})`:"(no known reference)"}`).join('  ')+
      (d.id==='point'&&!d.observed[index]?'\nObservation missing':'')+(!d.fit[index]?'\nNo fitted pose for this segment':'');
  }
}
function setupTimeSeries(){
  for(const id of ['plot-x','plot-y','plot-z','plot-w','plot-point'])el(id).addEventListener('change',drawTimeSeries);
  for(const id of ['video','chest-line','length','problem','displacement','translation','rotation','point']){
    const graph=el(id+'-series');
    new ResizeObserver(()=>{if(graph._fullLayout)Plotly.Plots.resize(graph);}).observe(graph);
    el(id+'-collapse').onclick=()=>{
      const panel=el(id+'-panel'),collapsed=!panel.classList.contains('collapsed-panel');
      panel.classList.toggle('collapsed-panel',collapsed);
      panel.classList.remove('expanded-plot');el(id+'-expand').textContent='Expand';
      el(id+'-collapse').textContent=collapsed?'Show':'Collapse';
      el(id+'-collapse').setAttribute('aria-expanded',String(!collapsed));
    };
    el(id+'-expand').onclick=()=>{
      el(id+'-panel').classList.remove('collapsed-panel');
      el(id+'-collapse').textContent='Collapse';el(id+'-collapse').setAttribute('aria-expanded','true');
      const panel=el(id+'-panel'),expand=!panel.classList.contains('expanded-plot');
      for(const other of ['video','chest-line','length','problem','displacement','translation','rotation','point']){el(other+'-panel').classList.remove('expanded-plot');el(other+'-expand').textContent='Expand';}
      panel.classList.toggle('expanded-plot',expand);el(id+'-expand').textContent=expand?'Restore':'Expand';
    };
  }
  document.addEventListener('keydown',e=>{if(e.key==='Escape')for(const id of ['video','chest-line','length','problem','displacement','translation','rotation','point']){el(id+'-panel').classList.remove('expanded-plot');el(id+'-expand').textContent='Expand';}});
  const splitter=el('plot-divider');let height=Math.round(innerHeight*.65);
  function size(value){height=Math.max(100,Math.min(Math.max(100,innerHeight-130),value));document.documentElement.style.setProperty('--plot-height',height+'px');splitter.setAttribute('aria-valuenow',Math.round(height));splitter.setAttribute('aria-valuemin',100);splitter.setAttribute('aria-valuemax',Math.max(100,innerHeight-130));}
  splitter.addEventListener('pointerdown',e=>{if(e.button!==0)return;splitter.setPointerCapture(e.pointerId);e.preventDefault();});
  splitter.addEventListener('pointermove',e=>{if(splitter.hasPointerCapture(e.pointerId))size(innerHeight-e.clientY-4);});
  splitter.addEventListener('pointerup',e=>{if(splitter.hasPointerCapture(e.pointerId))splitter.releasePointerCapture(e.pointerId);});
  splitter.addEventListener('keydown',e=>{if(['ArrowUp','ArrowDown'].includes(e.key)){e.preventDefault();size(height+(e.key==='ArrowUp'?20:-20));}});
  splitter.addEventListener('dblclick',()=>size(Math.round(innerHeight*.65)));window.addEventListener('resize',()=>size(height));size(height);
}
