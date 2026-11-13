"""Prepare a blank human development review packet; never invent judgments."""

from __future__ import annotations

import hashlib
import json
import os

import run  # noqa: F401
from multimodal_ecommerce.blind_evaluation import file_map, verify_files
from multimodal_ecommerce.safety import PROJECT_ROOT, write_json

ROOT = PROJECT_ROOT / "artifacts/module2_precision"
OUT = ROOT / "review_packet"
HTML = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>模块二开发相关性标注</title>
<style>body{font:16px system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:16px;color:#19324c;background:#f4f7fb}h1{font-size:25px}section,header{padding:20px;background:white;border:1px solid #d7e1eb;border-radius:10px;margin:12px 0}.images{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}img{width:100%;height:360px;object-fit:contain;background:#f5f5f5}button,input,textarea{font:inherit;padding:9px;margin:4px;border:1px solid #aebfd1;border-radius:6px}button{cursor:pointer;background:#ecf4ff}textarea{width:95%;min-height:60px}small{color:#53687d}.active{background:#b9dfc5}.danger{color:#8a3900}</style>
<h1>模块二开发相关性标注</h1><p class="danger">仅用于训练与开发。780对来自旧开发查询的Top30候选，不是独立验收集，也不能据此计算全图库Recall。</p>
<header><p>按查询主要商品的类型、用途及可见外观判断是否为相关商品。仅背景或人物相似不算相关；看不清时记录原因，保持待定。不要猜不可见材质或规格。这里判断业务相关性，并非确认同一SKU。</p>
<label>评审代号 <input id="reviewer" placeholder="填写真实评审代号"></label><label><input type="checkbox" id="human">由我本人逐项判断</label><br>
<button onclick="save()">导出全部进度</button><label>恢复进度 <input id="restore" type="file" accept=".jsonl"></label><p id="status" role="status"></p></header>
<section><p id="counter"></p><div class="images"><figure><img id="query" alt="查询服饰图片"><figcaption>查询</figcaption></figure><figure><img id="candidate" alt="候选服饰图片"><figcaption>候选</figcaption></figure></div>
<p><button id="yes" onclick="judge(1)">相关</button><button id="no" onclick="judge(0)">不相关</button><button onclick="judge(null)">待定</button></p><label>判断依据或无法判断原因<textarea id="note" oninput="rows[pos].note=this.value;dirty=true"></textarea></label>
<p><button onclick="move(-1)">上一对</button><button onclick="move(1)">下一对</button><button onclick="nextPending()">下一个未完成</button></p><small>不显示模型分数或原排名；页面没有后台自动保存，请导出进度。</small></section>
<script>
const rows=__ROWS__; let pos=0;let dirty=false;
const el=id=>document.getElementById(id);
function render(){const r=rows[pos];el('query').src=r.query_image;el('candidate').src=r.candidate_image;el('note').value=r.note;el('counter').textContent=`第 ${pos+1}/${rows.length} 对，已判断 ${rows.filter(x=>x.grade!==null).length} 对`;el('yes').className=r.grade===1?'active':'';el('no').className=r.grade===0?'active':'';}
function judge(value){rows[pos].grade=value;dirty=true;render();}
function move(delta){pos=Math.max(0,Math.min(rows.length-1,pos+delta));render();}
function nextPending(){for(let i=1;i<=rows.length;i++){let j=(pos+i)%rows.length;if(rows[j].grade===null){pos=j;render();return;}}el('status').textContent='所有关系已判断，请导出。';}
function save(){const name=el('reviewer').value.trim();if(rows.some(x=>x.grade!==null||x.note)&&(!name||!el('human').checked)){el('status').textContent='请填写评审代号，并确认由本人逐项判断。';return;}
const data=rows.map(r=>({...r,reviewer:r.grade!==null||r.note?name:'',review_origin:r.grade!==null||r.note?'human':null}));
const blob=new Blob([data.map(r=>JSON.stringify(r)).join('\n')+'\n'],{type:'application/x-ndjson;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='precision_development.reviewed.jsonl';a.click();URL.revokeObjectURL(a.href);dirty=false;}
el('restore').addEventListener('change',async ev=>{try{const parsed=(await ev.target.files[0].text()).trim().split(/\r?\n/).map(x=>JSON.parse(x));const expected=new Map(rows.map(r=>[r.id,r]));const seen=new Set();if(parsed.length!==rows.length)throw Error('条目数不符');for(const r of parsed){const original=expected.get(r.id);if(!original||seen.has(r.id)||r.packet_id!==original.packet_id||r.query_id!==original.query_id||r.candidate_id!==original.candidate_id||![0,1,null].includes(r.grade))throw Error('错包、重复或评分无效');seen.add(r.id);}for(const r of parsed){const target=expected.get(r.id);target.grade=r.grade;target.note=r.note||'';}el('status').textContent='进度已恢复；导出前请确认本人判断声明。';render();}catch(err){el('status').textContent='恢复失败：'+err.message;}});
window.addEventListener('beforeunload',ev=>{if(dirty){ev.preventDefault();ev.returnValue='';}});render();
</script></html>"""


def main() -> None:
    """Run the command-line workflow defined by this module.

    Raises:
        FileExistsError: Preserve the original blank packet.

    Notes:
        Writes derived artifacts through (OUT / 'index.html').write_text, OUT.mkdir,
        write_json.
    """
    if OUT.exists():
        raise FileExistsError("Preserve the original blank packet")
    verify_files(
        ROOT,
        json.loads((ROOT / "pair_cache_manifest.json").read_text(encoding="utf-8"))[
            "files"
        ],
    )
    queue = json.loads((ROOT / "model_inputs.json").read_text(encoding="utf-8"))
    packet = hashlib.sha256(json.dumps(queue, sort_keys=True).encode()).hexdigest()
    rows = []
    for query in queue:
        for document in query["documents"]:
            identity = query["query_id"] + "|" + document["id"]
            rows.append(
                {
                    "id": hashlib.sha256(identity.encode()).hexdigest()[:24],
                    "packet_id": packet,
                    "role": "development_training_only",
                    "query_id": query["query_id"],
                    "candidate_id": document["id"],
                    "query_image": os.path.relpath(query["query_image"], OUT).replace(
                        "\\", "/"
                    ),
                    "candidate_image": os.path.relpath(
                        document["image_path"], OUT
                    ).replace("\\", "/"),
                    "grade": None,
                    "note": "",
                    "reviewer": "",
                    "review_origin": None,
                }
            )
    rows.sort(
        key=lambda r: hashlib.sha256(("20261022:" + r["id"]).encode()).hexdigest()
    )
    OUT.mkdir()
    with (OUT / "judgments.blank.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    (OUT / "index.html").write_text(HTML.replace("__ROWS__", data), encoding="utf-8")
    assert all(r["grade"] is None and r["review_origin"] is None for r in rows)
    write_json(
        OUT / "manifest.json",
        {
            "role": "development_training_only",
            "rows": len(rows),
            "human_labels_completed": 0,
            "selection": "Old development queries, frozen fusion Top30; hash shuffled, hidden scores/ranks",
            "formal_acceptance": False,
            "full_gallery_recall_computable": False,
            "files": file_map(OUT),
        },
    )
    print(
        json.dumps(
            {"rows": len(rows), "human_labels_completed": 0, "formal_acceptance": False}
        )
    )


if __name__ == "__main__":
    main()
