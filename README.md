# QAssistant
# BigAlpha 2026 AI 开放创新赛决赛作品

QAssistant 是一个面向金融研究场景的多智能体 CLI 工具。它可以运行自动化研报生成流水线，也可以从本地 PDF 研报中抽取策略逻辑并生成可读、可改的策略样例代码。

项目当前以命令行为主入口，不再提供前后端展示功能。

## 安装

```powershell
pip install -e .
```

安装后可使用：

```powershell
QAssistant --help
```

也可以不安装 console script，直接运行：

```powershell
python -m src.cli --help
```

## 环境变量

QAssistant 通过环境变量读取模型、搜索和可选宏观数据服务配置。请将 `.env.example` 复制为 `.env`，再把 `MY_*` 示例值替换为你自己的配置。

```text
DS_MODEL_NAME=MY_DS_MODEL_NAME
DS_BASE_URL=MY_DS_BASE_URL
DS_API_KEY=MY_DS_API_KEY

VLM_MODEL_NAME=MY_VLM_MODEL_NAME
VLM_BASE_URL=MY_VLM_BASE_URL
VLM_API_KEY=MY_VLM_API_KEY

EMBEDDING_MODEL_NAME=MY_EMBEDDING_MODEL_NAME
EMBEDDING_BASE_URL=MY_EMBEDDING_BASE_URL
EMBEDDING_API_KEY=MY_EMBEDDING_API_KEY

SERPER_API_KEY=MY_SERPER_API_KEY
BOCHAAI_API_KEY=MY_BOCHAAI_API_KEY
FRED_API_KEY=MY_FRED_API_KEY
```

环境变量名称是运行契约的一部分；示例值中的 `MY_*` 只是占位符。

## 常用命令

运行完整研报生成流水线：

```powershell
QAssistant report run --config my_config.yaml --resume --max-concurrent 3
```

从本地 PDF 研报生成策略样例代码：

```powershell
QAssistant reproduce --pdf-path report.pdf --config my_config.yaml --report-id sample
```

检查配置和任务文件：

```powershell
QAssistant config validate --config my_config.yaml --tasks-file tasks.json
```

列出当前配置对应的输出产物：

```powershell
QAssistant outputs list --config my_config.yaml --json
```

## 配置

默认配置文件是 `my_config.yaml`。核心字段包括：

```yaml
target_name: "示例公司"
stock_code: "000001"
target_type: "company"
output_dir: "./outputs/exp-v1"
language: "zh"

custom_collect_tasks:
  - "收集公司财务报表和股价数据"
custom_analysis_tasks:
  - "分析公司盈利能力和竞争格局"

llm_config_list:
  - model_name: "${DS_MODEL_NAME}"
    api_key: "${DS_API_KEY}"
    base_url: "${DS_BASE_URL}"
  - model_name: "${EMBEDDING_MODEL_NAME}"
    api_key: "${EMBEDDING_API_KEY}"
    base_url: "${EMBEDDING_BASE_URL}"
  - model_name: "${VLM_MODEL_NAME}"
    api_key: "${VLM_API_KEY}"
    base_url: "${VLM_BASE_URL}"
```

运行前建议先校验：

```powershell
QAssistant config validate --config my_config.yaml
```

## 研报复现

研报复现功能只处理本地 PDF 文件，第一版目标是生成“可读、可改、结构清晰”的策略样例代码，不拉取真实行情数据、不接入回测框架、不输出收益指标。

输出目录：

```text
{output_dir}/{target_name}/report_reproduction/{report_id}/
```

主要产物：

```text
parsed_report.md
strategy_brief.json
sample_strategy.py
README_strategy.md
manifest.json
```

## 项目结构

```text
src/
  agents/              # 多智能体实现
  config/              # 配置加载和默认配置
  memory/              # Memory 与 checkpoint 状态
  pipeline/            # CLI 可复用 runner
  tools/               # 数据、搜索、PDF 等工具
  utils/               # LLM、日志、异步桥等公共工具
tests/                 # 轻量单元测试和 smoke tests
run_report.py          # 主研报兼容入口
run_report_reproduction.py
example.md             # CLI 参数和用法示例
```

## 测试

```powershell
B:\Anaconda\python.exe -m pytest tests\test_cli.py tests\test_pipeline_runners.py tests\test_pdf_report_parser.py tests\test_report_reproduction_agent.py tests\test_prompt_smoke.py tests\test_tool_registry.py tests\test_deepsearch_agent_unit.py tests\test_async_bridge.py
```

轻量测试不会调用真实 LLM、真实网络或完整端到端研报生成。
