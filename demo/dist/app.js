'use strict';
const $ = (id) => document.getElementById(id);
const state = {catalog: null, frame: null, busy: false, pending: false, timer: null, revision: 0, upload: null, uploadRevision:0, scoreScale:1};
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
function busy(value, text='running the models') {state.busy = value; $('busy').firstChild.textContent = text; $('busy').hidden = !value; $('sweep').disabled = value; $('export').disabled = value||!state.frame;}
function requestBody(query) {return state.upload ? {query, image_data:state.upload.data} : query;}
async function chooseImage() {
  const revision=++state.uploadRevision;
  const file=$('local-image').files[0];if(!file)return;
  if(!['image/png','image/jpeg'].includes(file.type)||file.size>state.catalog.max_upload_bytes){showError(new Error('choose an rgb png or jpeg up to 16 mib'));return;}
  try {
    const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result));reader.onerror=()=>reject(new Error('the file could not be read'));reader.readAsDataURL(file);});
    if(revision!==state.uploadRevision)return;
    state.upload={data};$('image').disabled=true;$('clear-upload').hidden=false;$('upload-note').textContent=file.name+' · label unknown';$('sweep-grid').innerHTML='<p class="empty">input changed. run the sweep for this image.</p>';queue();
  } catch(error) {if(revision===state.uploadRevision)showError(error);}
}
function clearImage(){state.uploadRevision++;state.upload=null;$('local-image').value='';$('image').disabled=false;$('clear-upload').hidden=true;$('upload-note').textContent='png or jpeg · up to 16 mib. processed locally; uploads are held in memory.';$('sweep-grid').innerHTML='<p class="empty">input changed. run the sweep for this selection.</p>';queue();}
function queue() {state.revision++; updateControlText(); clearTimeout(state.timer); state.timer = setTimeout(loadFrame, 180);}
async function loadFrame() {
  if (state.busy) {state.pending = true; return;}
  const revision = state.revision; busy(true); $('error').hidden = true;
  try {const data = await api(state.upload?'/api/upload':'/api/frame', requestBody(query())); if (revision === state.revision) render(data); else state.pending = true;}
  catch (error) {state.frame=null;$('models').replaceChildren();$('interpretation').textContent='no prediction for this input. check the message and choose a supported image.';showError(error);} finally {busy(false); if (state.pending) {state.pending = false; loadFrame();}}
}
function fmt(value, digits=4) {return value === null ? 'unavailable' : Number(value).toFixed(digits);}
function fmtSigned(value, digits=4) {return value === null ? 'unavailable' : `${value>=0?'+':''}${fmt(value,digits)}`;}
function fmtMse(value) {return value === null ? 'unavailable' : Number(value).toExponential(2);}
function displayNumbers(model, scale) {
  const convert = (value, multiplier) => value === null ? null : value * multiplier;
  return {score:convert(model.score,scale), threshold:convert(model.threshold,scale), score_change:convert(model.score_change,scale), pair_mse:convert(model.pair_mse,scale*scale)};
}
function displayConversion() {return {score_multiplier:state.scoreScale, mse_multiplier:state.scoreScale**2, scope:'display units only; source predictions, flags, maps and standardized margins retained'};}
function render(frame) {
  state.frame = frame; $('original').src = frame.original;
  $('image-label').textContent=frame.label===0?'label: normal':'label: unknown';
  $('image-scope').textContent=frame.label===0?'one normal image. three trained controls. the threshold stays fixed.':'one uploaded image. three trained controls. the threshold stays fixed.';
  $('scope-note').textContent=frame.label===0?'these are normal validation images, including synthetic lighting changes. a flag is a false alarm on this known normal image. this screen does not measure held-out defect recall.':'this uploaded image has no verified label. the selected category models keep their normal-validation thresholds. a flag is not a verified defect, and this screen does not estimate accuracy or recall.';
  $('timing').textContent = `${frame.mode} · ${Math.round(frame.milliseconds)} ms / view`;
  $('clipping').textContent = `${(frame.newly_clipped_fraction*100).toFixed(2)}% newly clipped channel values`;
  $('models').replaceChildren(...frame.models.map((model, index) => {
    const card = document.createElement('article'); card.className = 'model-card';
    const margin = model.standardized_margin;
    const numbers = displayNumbers(model,state.scoreScale);
    const units = state.scoreScale===1?'':' / 16';
    const digits = state.scoreScale===1?4:5;
    card.innerHTML = `<span class="model-index">${model.method} / 0${index+1}</span><h2>${titles[index]}</h2><p class="method-description">${descriptions[index]}</p><div class="map-stack"><img class="input" alt="relit input image"><img class="heatmap" alt="raw anomaly map on a validation-fixed display scale"></div><span class="decision ${model.flagged?'flag':''}">${model.flagged?'false alarm':'below threshold'}</span><p class="score-value">${fmt(margin,2)}<small> σ</small></p><p class="score-caption">margin above frozen threshold<br>in normal-validation score std units</p><div class="gauge"><i style="left:${margin===null?50:Math.max(0,Math.min(100,50+margin*12.5))}%"></i></div><div class="gauge-label"><span>−4σ</span><span>threshold</span><span>+4σ</span></div><div class="numbers"><div><span>anomaly score${units}</span><b>${fmt(numbers.score,digits)}</b></div><div><span>frozen threshold${units}</span><b>${fmt(numbers.threshold,digits)}</b></div><div><span>change from reference${units}</span><b>${fmtSigned(numbers.score_change,digits)}</b></div><div><span>patch pair mse${state.scoreScale===1?'':' / 256'}</span><b>${fmtMse(numbers.pair_mse)}</b></div><div><span>calibration</span><b>${model.calibration.validation_false_positives}/${model.calibration.validation_images} flagged</b></div></div>`;
    card.querySelector('.input').src = frame.altered; card.querySelector('.heatmap').src = model.heatmap; return card;
  }));
  document.documentElement.style.setProperty('--overlay', $('opacity').value);
  $('units-note').textContent = state.scoreScale===1 ? 'original score units. try dividing them by 16: the measured patch mse shrinks 256 times, with exactly the same decisions.' : 'scores and thresholds divided by 16; patch mse divided by 256. decisions, standardized margins and maps are unchanged. this is a change of units, not an improved detector.';
  const flagged = frame.models.filter(m=>m.flagged).map(m=>m.method);
  const flipped = frame.models.filter(m=>m.flagged!==m.reference_flagged).map(m=>m.method);
  $('interpretation').textContent = frame.label===null ? `${flagged.length?flagged.join(', ')+' exceeds its frozen threshold.':'all three models remain below their frozen thresholds.'} the uploaded image has no verified defect label.` : flagged.length ? `${flagged.join(', ')} flags this known normal image.${flipped.length?' '+flipped.join(', ')+' changes its decision from the reference light.':''} inspect the response; one image does not establish a population improvement.` : 'all three models remain below their own frozen thresholds on this normal image. compare the score changes, then inspect the fixed sweep.';
  if(frame.label===null)for(const decision of document.querySelectorAll('.decision.flag'))decision.textContent='flagged';
  const record = {...frame, display_conversion:displayConversion(), original:undefined, altered:undefined, models:frame.models.map(({heatmap,...rest})=>rest)};
  $('provenance').textContent = JSON.stringify(record, null, 2);
}
async function runSweep() {
  if (state.busy) return; const requested = query(); const revision = state.revision; busy(true,'measuring nine lighting settings'); $('error').hidden=true;
  try {const result = await api(state.upload?'/api/upload-sweep':'/api/sweep', requestBody(requested)); if (revision!==state.revision) {state.pending=true; return;}
    $('sweep-grid').replaceChildren(...result.rows[0].models.map((model,index)=>{
      const card=document.createElement('article');card.className='sweep-card';
      const title=document.createElement('h3');title.textContent=`${model.method} / ${titles[index]}`;card.append(title);
      const tiles=document.createElement('div');tiles.className='tiles';
      for(const row of result.rows){const m=row.models[index];const tile=document.createElement('button');tile.className=`tile ${m.flagged?'flag':''}`;tile.title=`brightness ${row.brightness}, gamma ${row.gamma}; raw score ${m.score}; threshold ${m.threshold}`;tile.innerHTML=`${m.flagged?'flagged':'below'}<span>b ${row.brightness.toFixed(1)} · γ ${row.gamma.toFixed(1)}</span>`;tile.addEventListener('click',()=>{$('brightness').value=row.brightness;$('gamma').value=row.gamma;queue();});tiles.append(tile);} card.append(tiles);return card;
    }));
  }catch(error){showError(error);}finally{busy(false);if(state.pending){state.pending=false;loadFrame();}}
}
$('export').addEventListener('click',()=>{if(!state.frame)return;const blob=new Blob([JSON.stringify({...state.frame,display_conversion:displayConversion()},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`illumination-${state.frame.query.category}-seed${state.frame.query.seed}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
$('score-units').addEventListener('change',()=>{state.scoreScale=Number($('score-units').value);if(state.frame)render(state.frame);});
$('category').addEventListener('change',categoryChanged);
$('choose-upload').addEventListener('click',()=>$('local-image').click());
$('local-image').addEventListener('change',chooseImage);
$('clear-upload').addEventListener('click',clearImage);
for(const id of ['seed','weight','image'])$(id).addEventListener('change',()=>{$('sweep-grid').innerHTML='<p class="empty">input changed. run the sweep for this selection.</p>';queue();});
for(const id of ['brightness','gamma'])$(id).addEventListener('input',queue);
$('opacity').addEventListener('input',()=>document.documentElement.style.setProperty('--overlay',$('opacity').value));
$('reset').addEventListener('click',()=>{$('brightness').value=1;$('gamma').value=1;queue();});
$('sweep').addEventListener('click',runSweep);
(async()=>{try{state.catalog=await api('/api/catalog');$('runtime').textContent=`● ${state.catalog.mode} / ${state.catalog.device}`;$('local-input').hidden=!state.catalog.accepts_uploads;for(const name of ['brightness','gamma'])$(name).step=String(state.catalog.lighting_step||.01);setOptions('category',Object.keys(state.catalog.categories),v=>v.replaceAll('_',' '));if(state.catalog.categories.fabric)$('category').value='fabric';categoryChanged();}catch(error){showError(error);$('runtime').textContent='runtime unavailable';}})();
