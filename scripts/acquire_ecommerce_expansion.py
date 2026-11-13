"""Download a bounded same-type e-commerce image batch for a new catalog version."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import ijson, httpx
from multimodal_ecommerce.safety import PROJECT_ROOT, ensure_output_path, sha256_file, write_json

SOURCE = PROJECT_ROOT / 'Alibaba阿里算法实习/多模态电商数据/product1m_product5m_train_id_label.json'
EXISTING = PROJECT_ROOT / 'artifacts/module2_p80_r90_20261106/development_benchmark/shared_gallery.json'
OUT = PROJECT_ROOT / 'data/module2_ecommerce_expansion_20261113'

def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=200);args=parser.parse_args()
    existing=json.loads(EXISTING.read_text()); used={r['id'].split('/')[-1] for r in existing['catalog']}; used_sha={r['image_sha256'] for r in existing['catalog']}
    media=ensure_output_path(OUT/'media');media.mkdir(parents=True,exist_ok=True)
    rows=[]; scanned=downloaded=failed=0
    with SOURCE.open('rb') as handle:
        for source_id,record in ijson.kvitems(handle,''):
            if len(rows)>=args.limit:break
            scanned+=1
            url=record.get('url'); title=record.get('title');
            if not source_id or source_id in used or not url or not title:continue
            ext='.jpg';dest=media/(str(source_id)+ext)
            try:
                if not dest.exists():
                    response=httpx.get(url,timeout=20,follow_redirects=True,headers={'User-Agent':'Mozilla/5.0'})
                    response.raise_for_status()
                    if not response.headers.get('content-type','').startswith('image/'):raise ValueError('not image')
                    dest.write_bytes(response.content)
                sha=sha256_file(dest)
                if sha in used_sha or any(r['image_sha256']==sha for r in rows):
                    dest.unlink(missing_ok=True);continue
                rows.append(dict(id='expanded/ecommerce/'+str(source_id),title=title,image_path=str(dest.relative_to(PROJECT_ROOT)),image_sha256=sha,source_id=str(source_id),source='product1m_train'))
                downloaded+=1
            except Exception:
                dest.unlink(missing_ok=True);failed+=1
    write_json(OUT/'catalog.json',rows);write_json(OUT/'receipt.json',dict(role='bounded_ecommerce_expansion_acquisition',source=str(SOURCE.relative_to(PROJECT_ROOT)),limit=args.limit,scanned=scanned,downloaded=downloaded,failed=failed,images=downloaded,models_run=False,active_platform_changed=False))
    print(json.dumps(dict(scanned=scanned,downloaded=downloaded,failed=failed),ensure_ascii=False))
if __name__=='__main__':main()
