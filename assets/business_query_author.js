/* Query authors see fixed scopes and query images, never model scores or catalog. */
function validateQueryExport(job,value){
  if(!value||Object.keys(value).sort().join('|')!==['release_id','role','reviewer','declared_human','rows'].sort().join('|')||value.release_id!==job.release_id||value.role!=='acceptance_query_author')throw Error('查询发布版本或格式不符');
  if(value.declared_human!==true||typeof value.reviewer!=='string'||!value.reviewer.trim())throw Error('需要本人署名与人工声明');
  if(!Array.isArray(value.rows)||value.rows.length!==job.rows.length)throw Error('不得删除固定查询');
  const editable=new Set(['text','relevance_rule','approved','reviewer','review_origin','note']),expected=new Map(job.rows.map(r=>[r.id,r])),seen=new Set();
  for(const r of value.rows){const t=expected.get(r.id);if(!t||seen.has(r.id))throw Error('未知或重复查询');seen.add(r.id);
    if(Object.keys(t).length!==Object.keys(r).length||Object.keys(t).some(k=>!(k in r)||(!editable.has(k)&&JSON.stringify(t[k])!==JSON.stringify(r[k]))))throw Error('查询范围或图片被改动');
    if(typeof r.approved!=='boolean'||['relevance_rule','reviewer','note'].some(k=>typeof r[k]!=='string')||![null,'human'].includes(r.review_origin))throw Error('确认字段无效');
    if(r.kind==='image'&&r.text!==null)throw Error('图片查询不能附加标题提示');
    if(r.kind==='text'&&r.text!==null&&typeof r.text!=='string')throw Error('查询文本类型无效');
    if([...editable].some(k=>r[k]!==t[k])&&(r.review_origin!=='human'||r.reviewer.trim()!==value.reviewer.trim()))throw Error('编写记录需要同一真人署名');
    if(r.approved&&(!r.relevance_rule.trim()||(r.kind==='text'&&!(r.text||'').trim())))throw Error('批准前请填写查询与相关性标准');
  }return value;
}
if(typeof module!=='undefined')module.exports={validateQueryExport};
if(typeof document!=='undefined'){
  const job=window.QUERY_JOB,$=id=>document.getElementById(id),key='business-query-author:'+job.release_id;
  let rows=job.rows.map(r=>({...r})),dirty=false;
  const signature=()=>{const name=$('reviewer').value.trim();if(!name||!$('human').checked)throw Error('请署名并确认本人编写');return name;};
  const value=()=>validateQueryExport(job,{release_id:job.release_id,role:'acceptance_query_author',reviewer:signature(),declared_human:true,rows});
  function progress(){$('progress').textContent=`已人工批准 ${rows.filter(r=>r.approved).length}/${rows.length}；未确认项仍待处理，相关性标签仍为0。`;}
  function save(){dirty=true;progress();try{localStorage.setItem(key,JSON.stringify(value()));$('status').textContent='已缓存本浏览器进度；交回前请导出JSON。';}catch(e){$('status').textContent='请导出保存：'+e.message;}}
  function stamp(row){row.reviewer=signature();row.review_origin='human';}
  function render(){ $('grid').replaceChildren();progress();for(const row of rows.filter(r=>$('filter').value==='all'||r.kind===$('filter').value)){
    const card=document.createElement('article'),heading=document.createElement('h2');heading.textContent=row.id+' · '+row.fine_type_scope+' · '+(row.kind==='text'?'文本需求':'图片需求');card.append(heading);
    if(row.kind==='image'){const a=document.createElement('a');a.href=job.images[row.id];a.target='_blank';const img=document.createElement('img');img.src=a.href;img.alt='待确认的独立查询图片';a.append(img);card.append(a);}
    for(const [field,label] of (row.kind==='text'?[['text','中文购物需求'],['relevance_rule','相关/不相关的判断标准'],['note','补充或无法确认原因']]:[['relevance_rule','相关/不相关的判断标准'],['note','主体不符、看不清等原因']])){
      const p=document.createElement('p');p.textContent=label;card.append(p);const area=document.createElement('textarea');area.value=row[field]||'';area.onchange=()=>{try{stamp(row);row[field]=area.value;row.approved=false;save();render();}catch(e){area.value=row[field]||'';$('status').textContent=e.message;}};card.append(area);
    }
    const label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.checked=row.approved;check.onchange=()=>{try{stamp(row);row.approved=check.checked;value();save();}catch(e){row.approved=false;check.checked=false;$('status').textContent=e.message;}};label.append(check,document.createTextNode('我已检查需求、图片主体及范围，并批准此查询'));card.append(label);$('grid').append(card);
  }}
  function restore(v){validateQueryExport(job,v);rows=v.rows.map(r=>({...r}));$('reviewer').value=v.reviewer;$('human').checked=true;render();}
  $('filter').onchange=render;
  $('export').onclick=()=>{try{const blob=new Blob([JSON.stringify(value(),null,2)+'\n'],{type:'application/json;charset=utf-8'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='acceptance_queries.reviewed.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);dirty=false;$('status').textContent='已导出。未经批准的查询保留，不会自动跳过。';}catch(e){$('status').textContent=e.message;}};
  $('restore').onchange=async e=>{try{restore(JSON.parse(await e.target.files[0].text()));save();}catch(err){$('status').textContent='恢复失败：'+err.message;}};
  try{const cached=localStorage.getItem(key);if(cached)restore(JSON.parse(cached));}catch(e){$('status').textContent='本浏览器无可用缓存，请使用导出JSON恢复。';}
  window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});render();
}
