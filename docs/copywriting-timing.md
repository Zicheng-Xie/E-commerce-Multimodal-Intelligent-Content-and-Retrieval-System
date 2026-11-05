# 本地分段耗时诊断

新增 `--profile-timing`，不改变提示词、商品事实或采样策略。
开启后在首个生成步结束同步一次CUDA，可能带来少量测量开销。
比较不同批次时，应保持该选项、模型、量化、缓存及输出上限一致。

## 五段指标

|阶段|文件或字段|口径|
|---|---|---|
|模型加载|loading.json: seconds|每个进程启动时一次；分列processor与权重加载、放置时间，不含Python导入|
|输入预处理|attempts[].timing.preprocess_seconds|prompt拼接、分词和同步后的GPU传输之和|
|TTFT|ttft_seconds|从generate调用前到第一个token选择完成；包含prefill与调用开销，不是纯forward时间|
|后续token解码|decode_seconds、decode_tokens_per_second|首token完成后到generate同步结束；速率分子为输出token数减1|
|后处理|postprocess_seconds|输出搬回CPU、文本解码、清洗、局部合并及JSON/规则校验，不含写文件|

当前输入只有文本，图片URL未下载或编码；image_count和image_encode_seconds均为0。
输出token统计包含特殊终止token（若生成），不是中文字数。

原 `generation_seconds` 与 `attempts[].seconds` 保留历史口径，未包含prompt拼接；
新增 `pipeline_seconds` 和 `attempt_seconds` 包含prompt拼接。
数据集读取、模型加载、记录写盘和报告写盘不计入单次请求。

## 推荐测试命令

项目根目录PowerShell执行：

```powershell
.\.venv-generation\Scripts\python.exe .\scripts\test_copywriting_batch.py run --samples '.\data\test data\raw_samples.json' --model-dir '.\models\Qwen3.5-4B' --quantization 4bit --compact-prompt --use-cache --max-tokens 384 --repair-attempts 0 --profile-timing
```

先关闭重试，定位首次生成的瓶颈。正式测整体业务耗时可恢复默认修复次数。
每条样本保存后自动更新 `timing_summary.json` 与 `timing_summary.md`。

重建报告（不加载模型）：

```powershell
.\.venv-generation\Scripts\python.exe .\scripts\test_copywriting_batch.py timing --run-dir '.\outputs\copywriting_v3\实际运行目录'
```

## 分组口径

- 使用分词后的真实input_tokens，不使用字符数替代。
- 固定token区间方便跨批次比较；若样本都在同一区间，再看短、中、长三等分组。
- 三等分组仅使用同一会话的成功、非首条、首次生成请求，报告实际token上下界。
- 同长度可能跨三等分边界；不要把分组名当作固定token范围。
- 初始生成、修复请求、模型启动后的第一条分开，避免混合统计。
- 同时观察实际输出token数和后续token/s，不能只比较请求总秒数。
- 保存mean、p50、p95及有效样本数；没有TTFT的旧记录保持缺失。

## 本次小样本诊断

已先扫描60条实际prompt：733～885 token，按长度排序各20条：
短733～767，中768～798，长798～885。当前没有图片输入长度差异。
从三层各选2条，交错顺序运行；固定128输出token上限、关闭重试。
这是计算性能诊断，文案可能截断，不能用于质量验收或估计完整384上限请求耗时。
首条单独标记冷启动；其余结果仍是小样本，不足以证明某个硬件原因。
