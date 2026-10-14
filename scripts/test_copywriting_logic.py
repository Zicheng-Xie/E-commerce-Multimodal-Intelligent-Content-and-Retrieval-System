import importlib.util
import json
import tempfile
import unittest
import sys
from contextlib import nullcontext
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

path=Path(__file__).with_name('test_copywriting_batch.py')
spec=importlib.util.spec_from_file_location('generation_v3',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def record(label,pv,title=None):
    return dict(label=label,pv=pv,title=title or label,url='',video='')
def args():
    return SimpleNamespace(style='',title_max=30,description_max=150,point_max=40,points=3)

class LogicTests(unittest.TestCase):
    def test_composition_order_alias_and_contradiction(self):
        sample=m.normalize_product('1',record('休闲裤','服装版型#:#直筒#;#裤长#:#长裤#;#材质成分#:#聚酯纤维65% 棉35%'))
        payload=m.make_payload(sample,args())
        out=dict(title='直筒休闲长裤',description='65%涤纶与35%棉混纺。',selling_points=['直筒设计，呈现直线裤腿轮廓。'])
        result,check=m.validate(json.dumps(out,ensure_ascii=False),payload)
        self.assertEqual(check['core_coverage'],1)
        self.assertTrue(check['fact_rule_pass'])
        out['description']='35%涤纶与65%棉混纺。'
        self.assertFalse(m.validate(json.dumps(out,ensure_ascii=False),payload)[1]['fact_rule_pass'])
        out['description']='100%纯棉，亲肤舒适。'
        self.assertFalse(m.validate(json.dumps(out,ensure_ascii=False),payload)[1]['fact_rule_pass'])

    def test_equal_units_dimensions_and_capacity_bound(self):
        facts={'商品类型':'水杯','尺寸':'直径46*高55mm','容量':'0.25L及以下'}
        payload=dict(facts=facts,core_fields=list(facts),length_limits={'title':60,'description':150,'selling_point':40},selling_point_count=3)
        out=dict(title='水杯容量≤250毫升',description='直径46毫米，高55毫米，容量不超过250毫升。',selling_points=['46毫米直径，核对摆放空间。'])
        check=m.validate(json.dumps(out,ensure_ascii=False),payload)[1]
        self.assertEqual(check['core_coverage'],1)
        self.assertTrue(check['fact_rule_pass'],check)
        out['title']='水杯容量250毫升'
        self.assertFalse(m.validate(json.dumps(out,ensure_ascii=False),payload)[1]['fact_rule_pass'])
        out['title']='水杯容量至少250毫升'
        self.assertFalse(m.validate(json.dumps(out,ensure_ascii=False),payload)[1]['fact_rule_pass'])

    def test_subcategory_color_and_extra_parameters(self):
        s=m.normalize_product('1',record('车载垃圾桶','颜色分类#:#黑色#;#颜色分类#:#白色'))
        self.assertEqual(s['facts']['商品类型'],'车载垃圾桶')
        self.assertEqual(s['options']['颜色'],['黑色','白色'])
        p=m.make_payload(s,args())
        out=dict(title='车载垃圾桶',description='不同容量可选。',selling_points=['用于车内垃圾收纳。'])
        self.assertFalse(m.validate(json.dumps(out,ensure_ascii=False),p)[1]['fact_rule_pass'])
        s=m.normalize_product('2',record('显示器','型号#:#M1#;#刷新率#:#144HZ#;#分辨率#:#2560x1440'))
        self.assertEqual(s['facts']['刷新率'],'144HZ')
        self.assertIn('分辨率',s['core_fields'])

    def test_effect_and_style_are_located(self):
        s=m.normalize_product('1',record('水杯','材质#:#不锈钢'))
        p=m.make_payload(s,args())
        out=dict(title='不锈钢水杯',description='采用简约设计，坚固耐用。',selling_points=['不锈钢材质，易清洁。'])
        c=m.validate(json.dumps(out,ensure_ascii=False),p)[1]
        self.assertFalse(c['fact_rule_pass'])
        self.assertIn('description',c['field_issues'])
        self.assertIn('selling_points',c['field_issues'])

    def test_partial_repair_keeps_untargeted_fields(self):
        previous=dict(title='原标题',description='旧详情',selling_points=['原卖点'])
        merged=json.loads(m.merge_partial(json.dumps({'description':'新详情'}),previous,['description']))
        self.assertEqual(merged['title'],'原标题')
        self.assertEqual(merged['selling_points'],['原卖点'])
        self.assertEqual(merged['description'],'新详情')
        self.assertEqual(m.merge_partial(json.dumps({'title':'偷改标题'}),previous,['description']),'')

    def test_format_failure_does_not_change_fluency_rating(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            p=Path(tmp)
            m.write_json(p/'manifest.json',{'samples':[dict(id='S001',category='3C数码')]})
            m.write_json(p/'review.json',{'items':[dict(id='S001',**{k:1 for k in m.REVIEW_FIELDS})]})
            m.write_json(p/'records/S001.json',{'result':{'title':'可读但超长的标题'},'validation':{'format_pass':False,'fact_rule_pass':True,'core_coverage':1}})
            report=m.summarize(p)
            self.assertEqual(report['overall']['fluency']['final_pass_rate'],1)
            self.assertEqual(report['delivery_pass_rate'],0)

    def test_resume_pending_preserves_existing_and_retry_is_explicit(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            p=Path(tmp)
            samples=[dict(id='S001',product_id='1'),dict(id='S002',product_id='2'),dict(id='S003',product_id='3')]
            m.write_json(p/'records/S001.json',{'sample':samples[0],'validation':{'quality_gate_pass':True}})
            m.write_json(p/'records/S002.json',{'sample':samples[1],'validation':{'quality_gate_pass':False}})
            before=(p/'records/S001.json').read_bytes()
            self.assertEqual(m.choose_pending(p,samples),[2])
            self.assertEqual(m.choose_pending(p,samples,True),[1,2])
            self.assertEqual((p/'records/S001.json').read_bytes(),before)
            m.write_json(p/'records/S001.json',{'sample':dict(product_id='wrong')})
            with self.assertRaises(ValueError):m.choose_pending(p,samples)

    def test_resume_completed_skips_model_and_keeps_review(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            p=Path(tmp)
            s=m.normalize_product('1',record('T恤','袖长#:#短袖'));s['id']='S001'
            a=args();a.run_dir=None;a.resume_dir=p;a.limit=0
            m.write_json(p/'manifest.json',{'samples':[s],'payloads':[m.make_payload(s,a)],'dry_run':False})
            m.write_json(p/'review.json',{'items':[dict(id='S001',**{k:None for k in m.REVIEW_FIELDS})]})
            r={'sample':s,'validation':{'format_pass':True,'fact_rule_pass':True,'core_coverage':1,'quality_gate_pass':True}}
            m.write_json(p/'records/S001.json',r)
            before=(p/'records/S001.json').read_bytes();review=(p/'review.json').read_bytes()
            m.run(a)
            self.assertEqual((p/'records/S001.json').read_bytes(),before)
            self.assertEqual((p/'review.json').read_bytes(),review)
            self.assertTrue((p/'summary.json').exists())

    def test_atomic_record_leaves_no_partial_final_file(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            p=Path(tmp)/'records/S001.json'
            m.atomic_record(p,{'sample':{'product_id':'1'},'result':{'title':'T恤'}})
            self.assertEqual(m.read_json(p)['result']['title'],'T恤')
            self.assertFalse(p.with_suffix('.json.tmp').exists())

    def test_parameter_list_and_option_claim_regression(self):
        s=m.normalize_product('1',record('显示器','品牌#:#Asus/华硕#;#型号#:#XG32VQ#;#接口#:#DP#;#尺寸#:#32英寸#;#颜色分类#:#黑色'))
        p=m.make_payload(s,args())
        self.assertFalse(p['has_options'])
        out=dict(title='Asus/华硕XG32VQ显示器',description='32英寸DP接口显示器，支持多种规格可选。',selling_points=['品牌型号：Asus/华硕XG32VQ','屏幕尺寸：32英寸','接口类型：DP'])
        _,checks=m.validate(json.dumps(out,ensure_ascii=False),p)
        self.assertTrue(checks['format_pass'])
        self.assertFalse(checks['copywriting_rule_pass'])
        self.assertFalse(checks['quality_gate_pass'])
        self.assertGreaterEqual(len(checks['copywriting_issues']),3)
        out['description']='Asus/华硕XG32VQ显示器采用32英寸屏幕，配备DP接口。'
        out['selling_points']=['32英寸屏幕，展开呈现画面内容。','DP接口，为设备视频连接提供接口。']
        self.assertTrue(m.validate(json.dumps(out,ensure_ascii=False),p)[1]['quality_gate_pass'])

    def test_run_three_categories_and_repair_without_model_weights(self):
        class Inputs(dict):
            def __init__(self,prompt):
                super().__init__(input_ids=SimpleNamespace(shape=(1,1)),prompt=prompt)
            def to(self,device):return self
        class Tensor:
            def __init__(self,text):self.text=text
            def __getitem__(self,key):return self
            def detach(self):return self
            def cpu(self):return self
            def tolist(self):return [ord(c) for c in self.text]
        class Tokenizer:
            eos_token_id=0
            def apply_chat_template(self,messages,**kwargs):return messages[-1]['content']
            def __call__(self,prompt,**kwargs):return Inputs(prompt)
            def decode(self,tokens,**kwargs):return ''.join(chr(t) for t in tokens)
        class Model:
            hf_device_map={}
            def eval(self):pass
            def get_input_embeddings(self):return SimpleNamespace(weight=SimpleNamespace(device='cpu'))
            def generate(self,**kwargs):
                request=json.loads(kwargs['prompt'])
                kind=request['facts']['商品类型']
                out=dict(title=kind,description='；'.join(request['facts'].values()),selling_points=[kind])
                if 'repair' not in request:out['description']+='防晒'
                if request.get('repair',{}).get('target_fields'):
                    out={k:out[k] for k in request['repair']['target_fields']}
                return Tensor(json.dumps(out,ensure_ascii=False))
        fake_torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,get_device_name=lambda _: 'fake',synchronize=lambda:None,empty_cache=lambda:None),bfloat16='fake',inference_mode=nullcontext,__version__='test')
        fake_transformers=SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=lambda *a,**k:SimpleNamespace(tokenizer=Tokenizer())),Qwen3_5ForConditionalGeneration=SimpleNamespace(from_pretrained=lambda *a,**k:Model()),__version__='test')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            p=Path(tmp);(p/'model').mkdir();(p/'model/config.json').write_text('{}')
            m.write_json(p/'raw.json',{'1':record('T恤','袖长#:#短袖'),'2':record('键盘','接口#:#USB'),'3':record('衣架','材质#:#塑料')})
            a=args();a.samples=p/'raw.json';a.limit=0;a.model_dir=p/'model';a.run_dir=p/'run';a.dry_run=False;a.repair_attempts=1;a.gpu_memory='5GiB';a.cpu_memory='10GiB';a.max_tokens=512;a.use_cache=False
            with patch.dict(sys.modules,{'torch':fake_torch,'transformers':fake_transformers}):m.run(a)
            for recpath in (p/'run/records').glob('*.json'):
                rec=m.read_json(recpath)
                self.assertEqual(len(rec['attempts']),2)
                self.assertTrue(rec['validation']['quality_gate_pass'])
            self.assertEqual(m.read_json(p/'run/summary.json')['acceptance']['status'],'待评测')
    def test_option_not_color_or_conflict(self):
        s=m.normalize_product('1',record('T恤','颜色分类#:#黑色M#;#颜色分类#:#白色L#;#尺码#:#M#;#尺码#:#L#;#材质#:#棉'))
        self.assertNotIn('颜色',s['facts'])
        self.assertEqual(s['options']['颜色分类'],['黑色M','白色L'])
        self.assertEqual(s['conflicts'],[])
        self.assertEqual(s['fashion_tags']['尺码']['value'],['M','L'])
    def test_brand_label_not_category(self):
        s=m.normalize_product('1',record('Dior/迪奥','领型#:#圆领','迪奥圆领T恤'))
        self.assertEqual(s['category'],'服饰穿搭')
        self.assertTrue(s['needs_review'])
        self.assertIsNone(m.normalize_product('2',record('资生堂','','资生堂精华')))
    def test_hub_not_keyboard(self):
        s=m.normalize_product('1',record('USB HUB/转换器','接口#:#USB','USB转换器支持键盘鼠标'))
        self.assertEqual(s['facts']['商品类型'],'USB HUB/转换器')
    def test_accessories_rejected(self):
        self.assertIsNone(m.normalize_product('1',record('平板电脑零部件','','平板维修键盘')))
        self.assertIsNone(m.normalize_product('2',record('鼠标/垫/贴','','鼠标垫')))
    def test_sparse_preserved(self):
        s=m.normalize_product('1',record('T恤','品牌#:#other/其他'))
        self.assertEqual(s['facts'],{'商品类型':'T恤'})
        self.assertEqual(s['core_fields'],['商品类型'])
    def test_conflicting_material_quarantined(self):
        s=m.normalize_product('1',record('衬衫','材质#:#棉混纺布#;#面料成分#:#棉100%'))
        self.assertNotIn('材质',s['facts'])
        self.assertNotIn('面料成分',s['facts'])
        self.assertEqual(s['fashion_tags']['材质']['status'],'待确认')
    def test_capacity_count_and_inconsistent_volume(self):
        s=m.normalize_product('1',record('衣架','容量#:#1个'))
        self.assertEqual(s['facts']['数量'],'1个')
        self.assertNotIn('容量',s['facts'])
        s=m.normalize_product('2',record('收纳箱','尺寸#:#32cmx22cmx16cm#;#容量#:#50L'))
        self.assertNotIn('容量',s['facts'])
        self.assertNotIn('尺寸',s['facts'])
    def test_pv_malformed_and_repeated(self):
        ap,issues=m.parse_pv('材质#:#棉#;#材质#:#棉#;#坏片段#;#品牌#:#')
        self.assertEqual(ap,{'材质':['棉']})
        self.assertEqual(len(issues),2)
    def test_style_override_and_duplicate_products(self):
        s=m.normalize_product('1',record('T恤','袖长#:#短袖'))
        s['style']='复古'
        a=args();a.style='专业简洁'
        self.assertEqual(m.make_payload(s,a)['style'],'专业简洁')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path=Path(tmp)/'samples.json'
            m.write_json(path,[dict(s,id='S001'),dict(s,id='S002')])
            with self.assertRaises(ValueError):m.load_samples(path)
    def test_model_validate_claims_coverage_and_lengths(self):
        s=m.normalize_product('1',record('移动电源','型号#:#P20#;#容量#:#20000mAh'))
        p=m.make_payload(s,args())
        good=dict(title='P20移动电源',description='容量20000mAh。',selling_points=['20000mAh容量'])
        _,c=m.validate(json.dumps(good),p)
        self.assertTrue(c['quality_gate_pass'])
        good['description']='容量30000mAh，支持快充。'
        _,c=m.validate(json.dumps(good),p)
        self.assertFalse(c['quality_gate_pass'])
        self.assertTrue(c['fact_risks'])
        good['title']='长'*40
        _,c=m.validate(json.dumps(good),p)
        self.assertFalse(c['format_pass'])
        self.assertIsNone(m.validate('not json',p)[0])
    def test_acceptance_missing_or_failed_not_pass(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            folder=Path(tmp)
            samples=[dict(id='S001',category='服饰穿搭'),dict(id='S002',category='3C数码')]
            m.write_json(folder/'manifest.json',{'samples':samples})
            m.write_json(folder/'review.json',{'items':[dict(id=s['id'],**{k:1 for k in m.REVIEW_FIELDS}) for s in samples]})
            r=dict(validation=dict(format_pass=True,fact_rule_pass=True,quality_gate_pass=True,core_coverage=1))
            m.write_json(folder/'records/S001.json',r)
            self.assertEqual(m.summarize(folder)['acceptance']['status'],'待评测')
            r['error']='OOM'
            m.write_json(folder/'records/S002.json',r)
            report=m.summarize(folder)
            self.assertEqual(report['acceptance']['relevance_rate'],0.5)
            self.assertEqual(report['acceptance']['status'],'未通过')
    def test_unreviewed_and_factuality_in_relevance(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            folder=Path(tmp)
            m.write_json(folder/'manifest.json',{'samples':[dict(id='S001',category='3C数码')]})
            m.write_json(folder/'records/S001.json',{'validation':{'format_pass':True,'fact_rule_pass':True,'core_coverage':1}})
            review=dict(id='S001',**{k:None for k in m.REVIEW_FIELDS})
            m.write_json(folder/'review.json',{'items':[review]})
            self.assertEqual(m.summarize(folder)['acceptance']['status'],'待评测')
            review.update({k:1 for k in m.REVIEW_FIELDS});review['factuality']=0
            m.write_json(folder/'review.json',{'items':[review]})
            self.assertEqual(m.summarize(folder)['acceptance']['relevance_rate'],0)

if __name__=='__main__':unittest.main()
