/* Catalog usability QA only: never creates retrieval relevance labels. */
function validateCatalog(job,value){
  const keys=['release_id','review_slot','reviewer','declared_human','rows'];
  if(!value||Object.keys(value).sort().join('|')!==keys.sort().join('|')||value.release_id!==job.release_id||value.review_slot!==job.review_slot)throw Error('发布版本或席位不符');
  if(value.declared_human!==true||typeof value.reviewer!=='string'||!value.reviewer.trim())throw Error('需要本人署名及人工声明');
  if(!Array.isArray(value.rows)||value.rows.length!==job.rows.length)throw Error('条目数量不符');
  const expected=new Map(job.rows.map(r=>[r.id,r])),seen=new Set(),editable=new Set(['verdict','product_type','note','reviewer','review_origin']);
  const norm=s=>s.normalize('NFKC').trim().toLowerCase();
  for(const row of value.rows){
    const source=expected.get(row.id);if(!source||seen.has(row.id))throw Error('未知或重复条目');seen.add(row.id);
    if(Object.keys(source).length!==Object.keys(row).length||Object.keys(source).some(k=>!(k in row)||!editable.has(k)&&JSON.stringify(source[k])!==JSON.stringify(row[k])))throw Error('原始商品信息被修改');
    if(![null,'usable','reject','uncertain'].includes(row.verdict)||row.role!=='development_catalog_QA_only')throw Error('不是商品质检判断');
    if(!['product_type','note','reviewer'].every(k=>typeof row[k]==='string')||![null,'human'].includes(row.review_origin))throw Error('署名或文本格式不符');
    if((row.verdict!==null||row.product_type.trim()||row.note.trim())&&(row.review_origin!=='human'||norm(row.reviewer)!==norm(value.reviewer)))throw Error('逐项判断须保留本人署名');
    if(row.verdict==='usable'&&!row.product_type.trim())throw Error('可用商品请填写实际商品主体');
    if(['reject','uncertain'].includes(row.verdict)&&!row.note.trim())throw Error('不适合或待核实请填写理由');
  }
  return value;
}
if(typeof module!=='undefined')module.exports={validateCatalog};
if(typeof document!=='undefined'){
 const job=window.JOB,$=id=>document.getElementById(id),cache='catalog-qa:'+job.release_id+':'+job.review_slot;
 let rows=job.rows.map(r=>({...r})),page=0;
 const status=s=>$('status').textContent=s;
 const signature=()=>{const n=$('reviewer').value.trim();if(!n||!$('human').checked)throw Error('先填写本人代号并确认逐项判断');return n;};
 const submission=()=>validateCatalog(job,{release_id:job.release_id,review_slot:job.review_slot,reviewer:signature(),declared_human:true,rows});
 function save(){try{localStorage.setItem(cache,JSON.stringify(submission()));status('已保存本浏览器进度，交回前请导出JSON。');}catch(e){status(e.message);}}
 function render(){
  $('heading').textContent='新增开发商品质检 · 席位'+job.review_slot;
  $('progress').textContent=`第${page+1}/${Math.ceil(rows.length/8)}页 · 已处理${rows.filter(r=>r.verdict!==null).length}/${rows.length}件`;
  $('grid').replaceChildren();
  for(const r of rows.slice(page*8,page*8+8)){
   const card=document.createElement('article'),a=document.createElement('a'),img=document.createElement('img');a.href=r.image;a.target='_blank';a.rel='noopener';img.src=r.image;img.alt='商品原图，点击查看大图';a.append(img);card.append(a);
   for(const text of [r.title,'待核实方向：'+r.bucket_name,'来源元数据细类：'+r.source_label]){const p=document.createElement('p');p.textContent=text;card.append(p);}
   const type=document.createElement('input');type.placeholder='观察到的商品主体，例如无线耳机本体';type.value=r.product_type;type.onchange=()=>{try{r.reviewer=signature();r.review_origin='human';r.product_type=type.value;save();}catch(e){status(e.message);}};card.append(type);
   const note=document.createElement('input');note.placeholder='不适合、待核实或重复风险的理由';note.value=r.note;note.onchange=()=>{try{r.reviewer=signature();r.review_origin='human';r.note=note.value;save();}catch(e){status(e.message);}};card.append(note);
   for(const [label,value]of [['可用','usable'],['不适合','reject'],['待核实','uncertain'],['未处理',null]]){const b=document.createElement('button');b.textContent=label;b.className=r.verdict===value?'chosen':'';b.onclick=()=>{try{r.reviewer=signature();r.review_origin='human';r.verdict=value;save();render();}catch(e){status(e.message);}};card.append(b);}
   $('grid').append(card);
  }
  $('prev').disabled=page===0;$('next').disabled=(page+1)*8>=rows.length;
 }
 function restore(v){validateCatalog(job,v);rows=v.rows.map(r=>({...r}));$('reviewer').value=v.reviewer;$('human').checked=true;render();}
 $('prev').onclick=()=>{page--;render();};$('next').onclick=()=>{page++;render();};
 $('export').onclick=()=>{try{const blob=new Blob([JSON.stringify(submission(),null,2)+'\n'],{type:'application/json;charset=utf-8'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='catalog-qa.'+job.review_slot+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);status('已导出；此文件仅含商品质检判断，不是检索相关性标签。');}catch(e){status(e.message);}};
 $('restore').onchange=async e=>{try{restore(JSON.parse(await e.target.files[0].text()));save();}catch(err){status(err.message);}};
 try{const old=localStorage.getItem(cache);if(old)restore(JSON.parse(old));}catch(e){status('缓存无法恢复，请使用之前导出的文件。');}render();
}
