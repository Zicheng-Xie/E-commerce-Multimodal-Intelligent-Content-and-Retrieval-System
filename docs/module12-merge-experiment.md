# 模块 1 + 模块 2 合并实验

基于 module2-dev b2941f3，合入 shz-dev 61d0c16；共同祖先 19e5f87。

## 分支差异

模块 1 独有 6 个提交，模块 2 独有 3 个提交。提交数量不等于失败数量。

存在不同内容的共同文件为 .flake8、.gitignore、README.md、pyproject.toml。README 属于双方内容修改；.flake8 和 pyproject.toml 属于双方新增。首次实验未执行真实 merge；2026-11-16 已补齐并验证真实 Git 对象 SHA，执行 git merge --no-commit --no-ff 后确认：4 个冲突文件、5 个冲突块（.flake8 1 块，.gitignore 1 块，README.md 1 块，pyproject.toml 2 块）。前三个配置冲突保留模块 2 版本；README 使用实验说明，模块 1 工具配置独立保存。

## 解决方案

- 保留模块 2 的源码、依赖、根 pyproject.toml、.flake8 和 .gitignore。
- 原样接入模块 1 的生成脚本、兼容测试入口、183 项测试、开发依赖清单与编码规范。
- 将模块 1 工具配置保存在 configs/copywriting-quality.toml 与 configs/copywriting.flake8；调整其 CI 显式读取配置。
- 模块 1 生成依赖为 Python 3.11 / Transformers 5.3.0，模块 2 使用 Python 3.12 / Transformers 4.57.1；分别建立环境，避免强行合成一份模型依赖。

## 验证

模块 1：python scripts/test_copywriting_logic.py，183 项测试通过（本次实际运行，无模型）。模块 2 的源文件在云端基准树中保持，未运行真实模型生成或网页集成测试。现有模块 2 基准此前 456 项测试通过，本次不把历史结果作为重新运行成绩。

## 下一步：网页入口

模块 2 展示页中增加“商品文案生成”入口，后端通过单独生成环境启动有参数白名单的本地任务；先实现输入校验、任务状态和结果展示，再验证 GPU 资源互斥。当前实验只完成源码共存，不宣称网页功能已打通。

## 2026-11-16 实际合并复核

在隔离目录执行真实合并，冲突已全部标记解决。模块 1 独立测试 183 项通过。全量回归首次发现模块 2 云端缺少 assets/business_query_author.js、business_review_jobs.js、catalog_qa.js，造成 3 项失败；本次从本地已有源码补齐这些文件，未改动其逻辑。补齐后全量 639 项通过，另 54 个 subtests 通过。未执行真实模型生成或网页新增入口验收。
