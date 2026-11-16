/* Human-only offline workbench; exports keep original immutable rows. */
function validateSubmission(job, value) {
  const keys=['release_id','batch_id','review_slot','reviewer','declared_human','adjudication_of','rows'];
  if(!value||Object.keys(value).sort().join('|')!==keys.sort().join('|'))throw Error('导出格式不符');
  for(const k of ['release_id','batch_id','review_slot','adjudication_of'])if(value[k]!==job[k])throw Error('发布版本、批次或席位不符');
  if(value.declared_human!==true||typeof value.reviewer!=='string'||!value.reviewer.trim())throw Error('需要本人署名及人工声明');
  if(!Array.isArray(value.rows)||value.rows.length!==job.rows.length)throw Error('条目数不符');
  const expected=new Map(job.rows.map(r=>[r.id,r])),seen=new Set(),editable=new Set(['grade','note','reviewer','review_origin']);
  const norm=s=>s.normalize('NFKC').trim().toLowerCase();
  for(const r of value.rows){
    const t=expected.get(r.id);if(!t||seen.has(r.id))throw Error('未知或重复条目');seen.add(r.id);
    if(Object.keys(r).length!==Object.keys(t).length||Object.keys(t).some(k=>!(k in r)||(!editable.has(k)&&JSON.stringify(r[k])!==JSON.stringify(t[k]))))throw Error('原始条目被修改');
    if(!(r.grade===null||(Number.isInteger(r.grade)&&[0,1].includes(r.grade))))throw Error('标签只能为整数0、1或null');
    if(typeof r.note!=='string'||typeof r.reviewer!=='string'||![null,'human'].includes(r.review_origin))throw Error('署名或依据无效');
    if((r.grade!==null||r.note.trim())&&(r.review_origin!=='human'||norm(r.reviewer)!==norm(value.reviewer)))throw Error('存在不同评审人的条目，请恢复正确署名');
    if(job.review_slot==='adjudicator'&&r.grade!==null&&!r.note.trim())throw Error('仲裁必须逐项写出理由');
  }
  return value;
}
function makeSubmission(job,rows,reviewer,declared){return validateSubmission(job,{release_id:job.release_id,batch_id:job.batch_id,review_slot:job.review_slot,adjudication_of:job.adjudication_of,reviewer:reviewer.trim(),declared_human:declared,rows:rows.map(r=>({...r}))});}
if(typeof module!=='undefined')module.exports={validateSubmission,makeSubmission};
if(typeof document!=='undefined'){
  const job=window.JOB,$=id=>document.getElementById(id),storageKey='business-review:'+job.release_id+':'+job.batch_id+':'+job.review_slot+':'+(job.adjudication_of||'primary');
  let rows=job.rows.map(r=>({...r})),page=0,dirty=false;
  $('heading').textContent=`${job.review_slot==='adjudicator'?'第三人仲裁':'开发评审席位 '+job.review_slot} · ${job.batch_id}`;
  const message=s=>{$('status').textContent=s;};
  function signature(){const n=$('reviewer').value.trim();if(!n||!$('human').checked)throw Error('先填写本人唯一代号并勾选人工声明');return n;}
  function save(){dirty=true;try{localStorage.setItem(storageKey,JSON.stringify(makeSubmission(job,rows,signature(),true)));message('进度已保存到本浏览器；交回前仍须导出JSON。');}catch(e){message('请导出保存：'+e.message);}}
  function picture(src){const a=document.createElement('a');a.href=src;a.target='_blank';a.rel='noopener';const img=document.createElement('img');img.src=src;img.loading='lazy';img.alt='商品图片，点击查看原图';a.append(img);return a;}
  function render(){
    const first=rows[0];$('query').replaceChildren();
    const title=document.createElement('h2');title.textContent=first.query_text||'图片查询';$('query').append(title);
    const rule=document.createElement('p');rule.textContent='相关性标准：'+first.query_rule;$('query').append(rule);
    if(job.presentation[first.id].query_image)$('query').append(picture(job.presentation[first.id].query_image));
    $('progress').textContent=`第${page+1}/${Math.ceil(rows.length/12)}页 · 已判断${rows.filter(r=>r.grade!==null).length}/${rows.length}项`;
    $('grid').replaceChildren();
    for(const r of rows.slice(page*12,(page+1)*12)){
      const card=document.createElement('article');card.className='card '+(r.grade===1?'positive':r.grade===0?'negative':'');card.append(picture(job.presentation[r.id].candidate_image));
      const text=document.createElement('p');text.textContent=r.candidate_title||'候选商品';card.append(text);
      if(job.conflicts){const c=job.conflicts[r.id],why=document.createElement('p');why.textContent=`A：${c.A.grade}（${c.A.note||'无补充说明'}）；B：${c.B.grade}（${c.B.note||'无补充说明'}）`;card.append(why);}
      for(const [label,value]of [['相关',1],['不相关',0],['待定',null]]){const b=document.createElement('button');b.textContent=label;b.className=r.grade===value?'chosen':'';b.onclick=()=>{try{const n=signature();r.grade=value;r.reviewer=n;r.review_origin='human';save();render();}catch(e){message(e.message);}};card.append(b);}
      const note=document.createElement('input');note.placeholder=job.conflicts?'仲裁理由（必填）':'判断依据或无法判断原因';note.value=r.note;note.onchange=()=>{try{const n=signature();r.note=note.value;r.reviewer=n;r.review_origin='human';save();}catch(e){note.value=r.note;message(e.message);}};card.append(note);$('grid').append(card);
    }
    $('prev').disabled=page===0;$('next').disabled=(page+1)*12>=rows.length;
  }
  function restore(value){validateSubmission(job,value);rows=value.rows.map(r=>({...r}));$('reviewer').value=value.reviewer;$('human').checked=true;render();}
  $('prev').onclick=()=>{if(page>0)page--;render();};$('next').onclick=()=>{if((page+1)*12<rows.length)page++;render();};
  $('export').onclick=()=>{try{const value=makeSubmission(job,rows,signature(),$('human').checked),blob=new Blob([JSON.stringify(value,null,2)+'\n'],{type:'application/json;charset=utf-8'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=job.batch_id+'.'+job.review_slot+'.reviewed.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);dirty=false;message('已导出，未判断的关系仍为null。每批只交回最新一个文件。');}catch(e){message(e.message);}};
  $('restore').onchange=async e=>{try{restore(JSON.parse(await e.target.files[0].text()));save();message('已恢复并保留原署名；请确认是本人评审记录。');}catch(err){message('恢复失败：'+err.message);}};
  try{const saved=localStorage.getItem(storageKey);if(saved){restore(JSON.parse(saved));message('已恢复本浏览器的本批记录；请确认本人署名。');}}catch(e){message('浏览器缓存不可用或记录不匹配，请使用JSON导出恢复。');}
  window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});render();
}
