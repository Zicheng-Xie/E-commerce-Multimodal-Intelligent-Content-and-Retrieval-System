/* Offline development-label grid. No model scores or automatic relevance labels. */
function validateReviewRows(template, incoming) {
  if (!Array.isArray(incoming) || incoming.length !== template.length) throw Error('条目数不符');
  const expected = new Map(template.map(r => [r.id, r])); const seen = new Set();
  const editable = new Set(['grade','note','reviewer','review_origin']);
  for (const row of incoming) {
    const original = expected.get(row.id);
    if (!original || seen.has(row.id)) throw Error('未知或重复条目');
    seen.add(row.id);
    if (Object.keys(row).length !== Object.keys(original).length || Object.keys(original).some(k => !(k in row) || (!editable.has(k) && row[k] !== original[k]))) throw Error('原始条目或图片字段被修改');
    if (!(row.grade === null || (Number.isInteger(row.grade) && (row.grade === 0 || row.grade === 1)))) throw Error('标签必须为0、1或null');
    if (typeof row.note !== 'string' || typeof row.reviewer !== 'string' || ![null,'human'].includes(row.review_origin)) throw Error('评审声明无效');
    if ((row.grade !== null || row.note.trim()) && (!row.reviewer.trim() || row.review_origin !== 'human')) throw Error('判断必须包含人工评审声明');
  }
  return incoming.map(r => ({...r}));
}
function reviewRowsToJsonl(rows) { return rows.map(r => JSON.stringify(r)).join('\n') + '\n'; }
function markVisiblePending(rows, ids, reviewer) {
  if (typeof reviewer !== 'string' || !reviewer.trim()) throw Error('请填写评审代号');
  const visible = new Set(ids);
  return rows.map(r => visible.has(r.id) && r.grade === null ? {...r,grade:0,reviewer:reviewer.trim(),review_origin:'human'} : {...r});
}
if (typeof module !== 'undefined' && module.exports) module.exports={validateReviewRows,reviewRowsToJsonl,markVisiblePending};
if (typeof document !== 'undefined') {
  const template = window.REVIEW_TEMPLATE;
  let rows = template.map(r=>({...r})); let queryPos=0; let page=0; let dirty=false;
  const queries=[...new Set(rows.map(r=>r.query_id))]; const pageSize=12;
  const el=id=>document.getElementById(id);
  const signature=()=>{const name=el('reviewer').value.trim();if(!name||!el('human').checked)throw Error('请填写评审代号并确认本人逐项判断');return name;};
  const current=()=>rows.filter(r=>r.query_id===queries[queryPos]);
  const visible=()=>current().slice(page*pageSize,(page+1)*pageSize);
  function message(value){el('status').textContent=value;}
  function render(){
    el('querySelect').value=String(queryPos);const group=current();const parts=Math.ceil(group.length/pageSize);
    el('queryImage').hidden=!group[0].query_image;
    if(group[0].query_image)el('queryImage').src=window.REVIEW_IMAGE_PREFIX+group[0].query_image;
    if(el('queryText'))el('queryText').textContent=group[0].query_text||'';
    if(el('queryRule'))el('queryRule').textContent=group[0].query_rule||'';
    el('progress').textContent=`查询 ${queryPos+1}/${queries.length} · 第 ${page+1}/${parts} 页 · 本查询已判 ${group.filter(r=>r.grade!==null).length}/${group.length} · 总计 ${rows.filter(r=>r.grade!==null).length}/${rows.length}`;
    el('grid').replaceChildren();
    for(const row of visible()){
      const card=document.createElement('article');card.className='card '+(row.grade===1?'positive':row.grade===0?'negative':'');
      const picture=document.createElement('img');picture.src=window.REVIEW_IMAGE_PREFIX+row.candidate_image;picture.alt='候选商品';picture.loading='lazy';card.append(picture);
      const tools=document.createElement('div');
      for(const [label,value] of [['相关',1],['不相关',0],['待定',null]]){
        const button=document.createElement('button');button.textContent=label;button.className=row.grade===value?'chosen':'';
        button.onclick=()=>{try{const name=signature();row.grade=value;row.reviewer=name;row.review_origin='human';dirty=true;message('');render();}catch(err){message(err.message);}};
        tools.append(button);
      }
      card.append(tools);
      if(row.candidate_title){const title=document.createElement('p');title.textContent=row.candidate_title;card.append(title);}
      const note=document.createElement('input');note.placeholder='依据或无法判断原因';note.value=row.note;
      note.onchange=()=>{try{const name=signature();row.note=note.value;row.reviewer=name;row.review_origin='human';dirty=true;}catch(err){note.value=row.note;message(err.message);}};
      card.append(note);el('grid').append(card);
    }
  }
  queries.forEach((id,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=`查询 ${i+1}`;el('querySelect').append(option);});
  el('querySelect').onchange=()=>{queryPos=Number(el('querySelect').value);page=0;render();};
  el('prev').onclick=()=>{if(page>0)page--;else if(queryPos>0){queryPos--;page=Math.ceil(current().length/pageSize)-1;}render();};
  el('next').onclick=()=>{if((page+1)*pageSize<current().length)page++;else if(queryPos+1<queries.length){queryPos++;page=0;}render();};
  el('bulk').onclick=()=>{try{const name=signature();rows=markVisiblePending(rows,visible().map(r=>r.id),name);dirty=true;message('仅本页待定项已由你的明确操作标为不相关；其他页面未变。');render();}catch(err){message(err.message);}};
  el('export').onclick=()=>{try{validateReviewRows(template,rows);const blob=new Blob([reviewRowsToJsonl(rows)],{type:'application/x-ndjson;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='precision_development.reviewed.jsonl';a.click();URL.revokeObjectURL(a.href);dirty=false;message(`已导出完整${rows.length}行，待定项保持null。`);}catch(err){message(err.message);}};
  el('restore').onchange=async ev=>{try{const text=await ev.target.files[0].text();rows=validateReviewRows(template,text.trim().split(/\r?\n/).map(line=>JSON.parse(line)));dirty=false;message('已恢复标签与原评审署名。');render();}catch(err){message('恢复失败：'+err.message);}};
  window.addEventListener('beforeunload',ev=>{if(dirty){ev.preventDefault();ev.returnValue='';}});
  render();
}
