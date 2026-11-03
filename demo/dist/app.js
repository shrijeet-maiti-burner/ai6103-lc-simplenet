'use strict';
const $ = (id) => document.getElementById(id);
const state = {catalog: null, frame: null, busy: false, pending: false, timer: null, revision: 0};
const descriptions = ['identity pair · discrimination only', 'relit pair · discrimination only', 'relit pair · discrimination + score mse'];
const titles = ['baseline', 'lighting augmentation', 'consistency ablation'];
function setOptions(id, values, label = (v) => v) {
  $(id).replaceChildren(...values.map(v => {const o = document.createElement('option'); o.value = v; o.textContent = label(v); return o;}));
}
function query() {return {category: $('category').value, seed: Number($('seed').value), image_id: $('image').value, weight: Number($('weight').value), brightness: Number($('brightness').value), gamma: Number($('gamma').value)};}
function updateControlText() {for (const name of ['brightness', 'gamma']) $(name+'-value').textContent = Number($(name).value).toFixed(2);}
function categoryChanged() {
  const item = state.catalog.categories[$('category').value];
  setOptions('seed', item.seeds); setOptions('weight', item.weights, v => `λ = ${v}`);
  $('weight').value = item.weights.includes(.1) ? '.1' : item.weights[0];
  // Browser option values use their canonical number spelling.
  if (!$('weight').value) $('weight').value = String(item.weights.includes(.1) ? .1 : item.weights[0]);
  setOptions('image', item.images, v => v.split('/').pop());
  $('sweep-grid').innerHTML = '<p class="empty">run the fixed 3 × 3 lighting sweep to see where each model changes its decision.</p>';
  queue();
}
async function api(path, body) {
  const r = await fetch(path, body ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {});
  const data = await r.json(); if (!r.ok) throw new Error(data.error || `request failed (${r.status})`); return data;
}
function showError(error) {$('error').textContent = error.message; $('error').hidden = false;}
function busy(value, text='running the models') {state.busy = value; $('busy').firstChild.textContent = text; $('busy').hidden = !value; $('sweep').disabled = value;}
function queue() {state.revision++; updateControlText(); clearTimeout(state.timer); state.timer = setTimeout(loadFrame, 180);}
async function loadFrame() {
  if (state.busy) {state.pending = true; return;}
  const revision = state.revision; busy(true); $('error').hidden = true;
  try {const data = await api('/api/frame', query()); if (revision === state.revision) render(data); else state.pending = true;}
  catch (error) {showError(error);} finally {busy(false); if (state.pending) {state.pending = false; loadFrame();}}
}
function fmt(value, digits=4) {return value === null ? 'unavailable' : Number(value).toFixed(digits);}
function render(frame) {
  state.frame = frame; $('original').src = frame.original;
  $('timing').textContent = `${frame.mode} · ${Math.round(frame.milliseconds)} ms / view`;
  $('clipping').textContent = `${(frame.newly_clipped_fraction*100).toFixed(2)}% newly clipped channel values`;
  $('models').replaceChildren(...frame.models.map((model, index) => {
    const card = document.createElement('article'); card.className = 'model-card';
    const margin = model.standardized_margin;
    card.innerHTML = `<span class="model-index">${model.method} / 0${index+1}</span><h2>${titles[index]}</h2><p class="method-description">${descriptions[index]}</p><div class="map-stack"><img class="input" alt="relit normal image"><img class="heatmap" alt="raw anomaly map on a validation-fixed display scale"></div><span class="decision ${model.flagged?'flag':''}">${model.flagged?'false alarm':'below threshold'}</span><p class="score-value">${fmt(margin,2)}<small> σ</small></p><p class="score-caption">margin above frozen threshold<br>in normal-validation score std units</p><div class="gauge"><i style="left:${margin===null?50:Math.max(0,Math.min(100,50+margin*12.5))}%"></i></div><div class="gauge-label"><span>−4σ</span><span>threshold</span><span>+4σ</span></div><div class="numbers"><div><span>raw anomaly score</span><b>${fmt(model.score)}</b></div><div><span>frozen threshold</span><b>${fmt(model.threshold)}</b></div><div><span>change from reference</span><b>${model.score_change>=0?'+':''}${fmt(model.score_change)}</b></div><div><span>normal patch pair mse</span><b>${fmt(model.pair_mse,5)}</b></div><div><span>calibration</span><b>${model.calibration.validation_false_positives}/${model.calibration.validation_images} flagged</b></div></div>`;
    card.querySelector('.input').src = frame.altered; card.querySelector('.heatmap').src = model.heatmap; return card;
  }));
  document.documentElement.style.setProperty('--overlay', $('opacity').value);
  const flagged = frame.models.filter(m=>m.flagged).map(m=>m.method);
  const flipped = frame.models.filter(m=>m.flagged!==m.reference_flagged).map(m=>m.method);
  $('interpretation').textContent = flagged.length ? `${flagged.join(', ')} flags this known normal image.${flipped.length?' '+flipped.join(', ')+' changes its decision from the reference light.':''} inspect the response; one image does not establish a population improvement.` : 'all three models remain below their own frozen thresholds on this normal image. compare the score changes, then inspect the fixed sweep.';
  const record = {...frame, original:undefined, altered:undefined, models:frame.models.map(({heatmap,...rest})=>rest)};
  $('provenance').textContent = JSON.stringify(record, null, 2);
}
async function runSweep() {
  if (state.busy) return; const requested = query(); const revision = state.revision; busy(true,'measuring nine lighting settings'); $('error').hidden=true;
  try {const result = await api('/api/sweep', requested); if (revision!==state.revision) {state.pending=true; return;}
    $('sweep-grid').replaceChildren(...result.rows[0].models.map((model,index)=>{
      const card=document.createElement('article');card.className='sweep-card';
      const title=document.createElement('h3');title.textContent=`${model.method} / ${titles[index]}`;card.append(title);
      const tiles=document.createElement('div');tiles.className='tiles';
      for(const row of result.rows){const m=row.models[index];const tile=document.createElement('button');tile.className=`tile ${m.flagged?'flag':''}`;tile.title=`brightness ${row.brightness}, gamma ${row.gamma}; raw score ${m.score}; threshold ${m.threshold}`;tile.innerHTML=`${m.flagged?'flagged':'below'}<span>b ${row.brightness.toFixed(1)} · γ ${row.gamma.toFixed(1)}</span>`;tile.addEventListener('click',()=>{$('brightness').value=row.brightness;$('gamma').value=row.gamma;queue();});tiles.append(tile);} card.append(tiles);return card;
    }));
  }catch(error){showError(error);}finally{busy(false);if(state.pending){state.pending=false;loadFrame();}}
}
$('export').addEventListener('click',()=>{if(!state.frame)return;const blob=new Blob([JSON.stringify(state.frame,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`illumination-${state.frame.query.category}-seed${state.frame.query.seed}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
$('category').addEventListener('change',categoryChanged);
for(const id of ['seed','weight','image'])$(id).addEventListener('change',()=>{$('sweep-grid').innerHTML='<p class="empty">input changed. run the sweep for this selection.</p>';queue();});
for(const id of ['brightness','gamma'])$(id).addEventListener('input',queue);
$('opacity').addEventListener('input',()=>document.documentElement.style.setProperty('--overlay',$('opacity').value));
$('reset').addEventListener('click',()=>{$('brightness').value=1;$('gamma').value=1;queue();});
$('sweep').addEventListener('click',runSweep);
(async()=>{try{state.catalog=await api('/api/catalog');$('runtime').textContent=`● ${state.catalog.mode} / ${state.catalog.device}`;for(const name of ['brightness','gamma'])$(name).step=String(state.catalog.lighting_step||.01);setOptions('category',Object.keys(state.catalog.categories),v=>v.replaceAll('_',' '));if(state.catalog.categories.fabric)$('category').value='fabric';categoryChanged();}catch(error){showError(error);$('runtime').textContent='runtime unavailable';}})();
