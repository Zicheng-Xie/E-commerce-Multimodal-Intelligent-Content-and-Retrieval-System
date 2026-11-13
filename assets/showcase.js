'use strict';
const $ = id => document.getElementById(id);
let mode = 'text', imageToken = null, samples = [], sampleOffset = 0, page = 1, total = 0, selectedProduct = null, busy = false;

async function api(path, data) {
  const response = await fetch(path, data === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || '服务暂时不可用');
  return value;
}
function error(message) { $('error').textContent = message; $('error').hidden = !message; }
function switchMode(value) {
  mode = value;
  $('textBox').hidden = value !== 'text'; $('imageBox').hidden = value !== 'image';
  for (const [id, chosen] of [['textTab', value === 'text'], ['imageTab', value === 'image']]) {
    $(id).classList.toggle('selected', chosen); $(id).setAttribute('aria-selected', String(chosen));
  }
  $('queryHint').textContent = value === 'text' ? '原图与标题共同参与商品匹配' : '识别主要商品，降低人物与背景干扰';
}
function selectImage(token, url, name) {
  imageToken = token; switchMode('image'); $('previewImage').src = url; $('previewName').textContent = name;
  $('dropzone').hidden = true; $('imagePreview').hidden = false; error('');
}
function showSamples() {
  $('sampleImages').replaceChildren();
  for (let i = 0; i < Math.min(7, samples.length); i++) {
    const row = samples[(sampleOffset + i) % samples.length], button = document.createElement('button'), img = document.createElement('img'), name = document.createElement('span');
    img.src = row.image_url; img.alt = row.name; name.textContent = row.name;
    button.append(img, name); button.onclick = () => selectImage(row.token, row.image_url, row.name); $('sampleImages').append(button);
  }
}
function detail(product) {
  selectedProduct = product; $('detailImage').src = product.image_url; $('detailImage').alt = product.title;
  $('detailTitle').textContent = product.title; $('detailId').textContent = '商品编号 · ' + product.id.split('/').at(-1); $('productDialog').showModal();
}
function render(products) {
  $('cards').replaceChildren();
  for (const row of products) {
    const card = document.createElement('button'); card.className = 'card'; card.onclick = () => detail(row);
    const picture = document.createElement('div'); picture.className = 'card-image'; const image = document.createElement('img');
    image.src = row.image_url; image.alt = row.title; image.loading = 'lazy'; picture.append(image);
    if (row.rank) { const rank = document.createElement('span'); rank.className = 'rank'; rank.textContent = 'TOP ' + row.rank; picture.append(rank); }
    const body = document.createElement('div'); body.className = 'card-body'; const title = document.createElement('h3'); title.textContent = row.title; title.title = row.title;
    const meta = document.createElement('small'), id = document.createElement('span'), arrow = document.createElement('span');
    id.textContent = 'NO. ' + row.id.split('/').at(-1); arrow.textContent = '查看商品 ↗'; meta.append(id, arrow); body.append(title, meta); card.append(picture, body); $('cards').append(card);
  }
}
async function catalog() {
  const value = await api('/api/catalog?page=' + page + '&size=20'); total = value.total; render(value.products);
  $('pagination').hidden = false; $('pageLabel').textContent = page + ' / ' + Math.ceil(total / 20);
  $('previousPage').disabled = page <= 1; $('nextPage').disabled = page * 20 >= total;
  $('resultsTitle').textContent = $('catalogHeading').hidden ? '逛逛商品图库' : '全部商品';
  $('resultsMeta').textContent = '共 ' + total + ' 件商品 · 展示 ' + ((page - 1) * 20 + 1) + '–' + Math.min(page * 20, total) + ' 件';
}
function navigation(library) {
  $('searchArea').hidden = library; $('catalogHeading').hidden = !library; $('showAll').hidden = library;
  $('navSearch').classList.toggle('active', !library); $('navCatalog').classList.toggle('active', library);
  $('breadcrumb').textContent = library ? '商品图库' : '多模态检索';
}
async function upload(file) {
  if (!file) return;
  if (busy) return;
  if (file.size > 8 * 1024 * 1024) return error('图片应小于 8 MB');
  $('searchButton').disabled = true;
  try {
    const source = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = reject; reader.readAsDataURL(file); });
    const value = await api('/api/upload', {base64:source.split(',')[1]}); selectImage(value.token, value.image_url, file.name);
  } catch (e) { error(e.message); } finally { $('searchButton').disabled = false; }
}
async function search() {
  if (busy) return;
  error(''); const request = mode === 'text' ? {text:$('queryText').value.trim()} : {image_token:imageToken};
  if (mode === 'text' && !request.text) return error('先输入你想找的商品。');
  if (mode === 'image' && !imageToken) return error('先上传图片，或选择一个示例。');
  busy = true; $('searchButton').disabled = true; $('searchButton').textContent = '正在检索…'; $('progress').hidden = false;
  const started = Date.now(); $('progressMessage').textContent = '正在提交检索…';
  $('elapsed').textContent = '0 秒';
  const ticker = setInterval(() => { $('elapsed').textContent = Math.floor((Date.now() - started) / 1000) + ' 秒'; }, 1000);
  try {
    const submitted = await api('/api/search', request); let result;
    for (let attempt = 0; attempt < 900; attempt++) {
      const job = await api('/api/jobs/' + submitted.job_id);
      if (job.state === 'error') throw new Error(job.error);
      if (job.state === 'done') { result = job.result; break; }
      $('progressMessage').textContent = job.message;
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    if (!result) throw new Error('检索时间过长，请查看本地服务状态。');
    render(result.results); $('pagination').hidden = true;
    $('resultsTitle').textContent = mode === 'text' ? '「' + request.text + '」的检索结果' : '图片检索结果';
    $('resultsMeta').textContent = '搜索 ' + result.searched + ' 件商品 · 返回 ' + result.results.length + ' 件 · 用时 ' + result.processing_seconds.toFixed(2) + ' 秒';
    document.querySelector('.results-section').scrollIntoView({behavior:'smooth', block:'start'});
  } catch (e) { error(e.message); }
  finally { clearInterval(ticker); busy = false; $('searchButton').disabled = false; $('searchButton').textContent = '开始检索 ↗'; $('progress').hidden = true; }
}
$('textTab').onclick = () => switchMode('text'); $('imageTab').onclick = () => switchMode('image');
$('clearImage').onclick = () => { imageToken = null; $('imagePreview').hidden = true; $('dropzone').hidden = false; $('imageFile').value = ''; };
$('imageFile').onchange = event => upload(event.target.files[0]);
$('dropzone').ondragover = event => { event.preventDefault(); $('dropzone').classList.add('dragging'); };
$('dropzone').ondragleave = () => $('dropzone').classList.remove('dragging');
$('dropzone').ondrop = event => { event.preventDefault(); $('dropzone').classList.remove('dragging'); upload(event.dataTransfer.files[0]); };
$('searchButton').onclick = search;
$('queryText').onkeydown = event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); search(); } };
document.querySelectorAll('[data-query]').forEach(button => button.onclick = () => { switchMode('text'); $('queryText').value = button.dataset.query; search(); });
$('moreSamples').onclick = () => { sampleOffset += 7; showSamples(); };
$('showAll').onclick = $('navCatalog').onclick = async () => { navigation(true); page = 1; await catalog(); window.scrollTo({top:0, behavior:'smooth'}); };
$('navSearch').onclick = async () => { navigation(false); page = 1; await catalog(); };
$('previousPage').onclick = async () => { page--; await catalog(); };
$('nextPage').onclick = async () => { page++; await catalog(); };
$('closeDialog').onclick = () => $('productDialog').close();
$('searchSimilar').onclick = () => { $('productDialog').close(); navigation(false); selectImage(selectedProduct.image_url.split('/').at(-1), selectedProduct.image_url, '商品图片'); window.scrollTo({top:0, behavior:'smooth'}); search(); };
(async () => {
  try {
    const data = await api('/api/samples'); samples = data.images; $('productCount').textContent = data.products; $('navCount').textContent = data.products;
    $('sampleStrip').hidden = !samples.length; showSamples(); await catalog();
  } catch (e) { error('服务连接失败：' + e.message); }
})();
