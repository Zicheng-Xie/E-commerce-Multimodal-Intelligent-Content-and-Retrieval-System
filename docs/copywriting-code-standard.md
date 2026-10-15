# 商品文案生成代码规范整改

本次依据《编码规范》整理现有代码，不改变生成提示词、量化方式、
缓存设置、生成长度、修复次数和事实检查规则。

## 修改范围

- `scripts/test_copywriting_batch.py`：Black 88字符格式、导入排序、
  公共函数类型注解和 Google 风格文档、结构化输入输出类型。
- 基础属性解析和扩展规则通过明确的私有函数委托，移除重复函数名覆盖。
- `tests/unit/test_copywriting_logic.py`：正式测试实现，隔离临时文件、
  标准输出和环境变量。
- `scripts/test_copywriting_logic.py`：兼容原测试命令的入口。
- 增加样本ID校验，拒绝通过修改manifest访问records目录之外的文件。
- `.github/workflows/copywriting-quality.yml`：无需模型和GPU的提交检查。

## 本地检查

建议在独立开发工具环境安装依赖，避免影响生成模型环境：

```powershell
py -3.11 -m venv .venv-quality
.\.venv-quality\Scripts\python.exe -m pip install -r requirements-copywriting-dev.txt
.\.venv-quality\Scripts\python.exe -m black --check --workers 1 scripts/test_copywriting_batch.py scripts/test_copywriting_logic.py tests/unit/test_copywriting_logic.py
.\.venv-quality\Scripts\python.exe -m isort --check-only scripts/test_copywriting_batch.py scripts/test_copywriting_logic.py tests/unit/test_copywriting_logic.py
.\.venv-quality\Scripts\python.exe -m flake8 scripts/test_copywriting_batch.py scripts/test_copywriting_logic.py tests/unit/test_copywriting_logic.py
.\.venv-quality\Scripts\python.exe -m mypy
.\.venv-quality\Scripts\python.exe -m pylint scripts/test_copywriting_batch.py scripts/test_copywriting_logic.py tests/unit/test_copywriting_logic.py
.\.venv-quality\Scripts\python.exe scripts/test_copywriting_logic.py
```

生成命令仍使用原来的`.venv-generation`环境。

## 验证结果与适用边界

- 25项单元测试通过，其中2项覆盖新增的样本ID安全校验。
- 60条既有商品的规范化结果与整改前一致。
- 9条既有文案的校验结果与整改前一致。
- 提示词与品类知识常量逐值比较一致。
- Black、isort、Flake8、mypy和Pylint配置内检查通过。
- 字符行宽另行检查为不超过88；Flake8按规范建议忽略E501，避免中文显示宽度差异。

mypy对复杂报告元数据和外部模型API采用渐进类型；Pylint当前阻断
E/F级错误，设计及重构建议仍需评审。没有通过关闭所有类型检查来代替整改。

本轮未加载真实模型，因此不作生成性能改善或文案验收达标的结论。
上一版本的数值误报、同义匹配限制仍然保留，需在独立功能修复中处理。

规范的单文件500行是建议项。为保留现有独立脚本调用方式，本轮保留
生成入口的模块布局；后续可将品类知识、输入解析、校验和模型运行拆为包。
部分存量回归用例覆盖连续状态变化，尚可继续拆成更细的独立场景。

## 提交建议

仅暂存本次修改文件，不提交模型、数据集、工具环境、备份和运行结果。
`.venv-quality/`应加入本地忽略规则。

建议提交信息：

```text
refactor: standardize copywriting scripts and add quality checks
```

提交前检查暂存差异；合并前仍须完成规范要求的代码评审。
