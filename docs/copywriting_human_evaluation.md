# 模块一生成与人工评测

输入采用官方商品 JSON 格式。以商品 ID 为键，保留 title、label、pv、url、video。模型权重和商品数据在本地准备。输出包含 title、description、selling_points，每条结果在运行目录的 records 中。

已验证的生成环境为 Python 3.11、CUDA 版 PyTorch 2.9.1（cu128）、Transformers 5.3.0、Accelerate 1.15.0 和 bitsandbytes 0.50.2。按本机 CUDA 环境安装 PyTorch，另外安装其余依赖：

```sh
python -m pip install transformers==5.3.0 accelerate==1.15.0 bitsandbytes==0.50.2
python scripts/test_copywriting_batch.py run --samples path/to/products.json --model-dir models/Qwen3.5-4B --run-dir outputs/dev20 --quantization 4bit --compact-prompt --use-cache --limit 20
python scripts/show_copywriting_results.py outputs/dev20 --html outputs/dev20/results.html
```

标题优先沿用原题中有属性依据的片段，显示器标题保留已核实型号。材质编造、功能虚标由规则拦截，并保留在校验记录中供复核。

## 开发样本试评

先用20条开发样本验证流程。打开生成的人工评测网页，对照原始属性填写相关度、通顺度及理由，然后下载评分 JSON。

```sh
python scripts/evaluate_copywriting.py prepare --run-dir outputs/dev20 --out outputs/dev20_form --role development
python scripts/evaluate_copywriting.py summarize --template outputs/dev20_form/review_template.json --review path/to/human_review.json --out outputs/dev20_review.json
```

相关度按核心字段命中、关键遗漏和事实错误逐条判定。通顺度按标题、详情和卖点的语言、逻辑和电商表达判定。开发样本评分只用于校准评分流程。

## 冻结样本验收

使用已冻结清单，或首次从官方商品池分层抽取三个品类各10条。每个历史目录可用一个 --history-root 提供。正式评测需要保留历史排除清单和冻结文件校验值。

```sh
python scripts/evaluate_copywriting.py freeze --source path/to/official_pool.json --history-root path/to/used_history --out outputs/frozen30
python scripts/evaluate_copywriting.py ready --frozen-dir outputs/frozen30 --development-report outputs/dev20_review.json
python scripts/test_copywriting_batch.py run --samples outputs/frozen30/raw_samples_acceptance30.json --model-dir models/Qwen3.5-4B --run-dir outputs/acceptance30 --evaluation-role acceptance --exclude-samples outputs/frozen30/used_history.json --quantization 4bit --compact-prompt --use-cache
python scripts/evaluate_copywriting.py prepare --run-dir outputs/acceptance30 --out outputs/acceptance30_form --role acceptance --frozen-dir outputs/frozen30 --development-report outputs/dev20_review.json
python scripts/evaluate_copywriting.py summarize --template outputs/acceptance30_form/review_template.json --review path/to/acceptance_review.json --out outputs/acceptance30_review.json
```

相关度通过率至少80%，即30条至少24条通过。通顺度通过率至少85%，即至少26条通过。两个维度均按人工评分汇总，分母为冻结样本总数。缺评时保留待评状态，自动规则结果另作诊断。

## 代码检查

```sh
python -m pip install -r requirements-copywriting-dev.txt
python scripts/test_copywriting_logic.py
```
