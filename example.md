# QAssistant CLI 使用示例

`QAssistant` 是 FinSight 项目的命令行入口，用于在本地运行多智能体金融研报生成和研报复现工作流。前后端 `demo/` 仍保留为 legacy demo，但日常使用推荐通过 CLI 执行。

## 安装

在项目根目录安装为可执行命令：

```powershell
pip install -e .
```

如果只想临时运行，也可以直接使用模块入口：

```powershell
python -m src.cli --help
```

## 环境变量

主研报流水线默认使用以下环境变量选择模型：

```powershell
$env:DS_MODEL_NAME="deepseek-chat"
$env:DS_API_KEY="your_api_key"
$env:DS_BASE_URL="https://api.deepseek.com"
$env:VLM_MODEL_NAME="your_vlm_model"
$env:VLM_API_KEY="your_vlm_key"
$env:VLM_BASE_URL="your_vlm_base_url"
$env:EMBEDDING_MODEL_NAME="your_embedding_model"
$env:EMBEDDING_API_KEY="your_embedding_key"
$env:EMBEDDING_BASE_URL="your_embedding_base_url"
```

也可以把这些变量写入项目根目录 `.env`。

## 常用命令

运行完整研报生成流水线：

```powershell
QAssistant report run --config my_config.yaml --resume --max-concurrent 3
```

从本地 PDF 研报生成策略样例代码：

```powershell
QAssistant reproduce --pdf-path report.pdf --config my_config.yaml --report-id demo
```

检查配置和任务文件：

```powershell
QAssistant config validate --config my_config.yaml --tasks-file tasks.json
```

列出当前配置对应的输出产物：

```powershell
QAssistant outputs list --config my_config.yaml --json
```

## 参数说明

### `report run`

| 参数 | 说明 |
| --- | --- |
| `--config` | 配置文件路径，默认 `my_config.yaml`。 |
| `--tasks-file` | 可选任务文件，支持 demo 格式 `collect_tasks/analysis_tasks`，也支持简化格式 `custom_collect_tasks/custom_analysis_tasks`。 |
| `--resume` | 从已有 Memory/checkpoint 继续执行。 |
| `--max-concurrent` | 同一阶段最多并发运行的 agent 数量。 |
| `--model-name` | 覆盖文本生成模型名，默认读取 `DS_MODEL_NAME`。 |
| `--vlm-model-name` | 覆盖视觉模型名，默认读取 `VLM_MODEL_NAME`。 |
| `--embedding-model-name` | 覆盖 embedding 模型名，默认读取 `EMBEDDING_MODEL_NAME`。 |
| `--no-auto-tasks` | 不额外调用 LLM 生成任务，只使用配置或任务文件中的任务。 |
| `--json` | 以 JSON 输出运行摘要。 |

### `reproduce`

| 参数 | 说明 |
| --- | --- |
| `--pdf-path` | 本地 PDF 研报路径，必填。 |
| `--config` | 配置文件路径，默认 `my_config.yaml`。 |
| `--report-id` | 复现产物目录名，便于稳定复跑。 |
| `--model-name` | 覆盖 LLM 模型名，默认读取 `DS_MODEL_NAME`。 |
| `--max-pages` | 限制 PDF 解析页数，适合先快速试跑。 |
| `--json` | 以 JSON 输出运行摘要。 |

### 通用配置覆盖

以下参数可用于 `report run`、`reproduce` 和 `outputs list`：

| 参数 | 说明 |
| --- | --- |
| `--target-name` | 覆盖配置中的研究对象名称。 |
| `--stock-code` | 覆盖股票代码。 |
| `--target-type` | 覆盖研究对象类型。 |
| `--output-dir` | 覆盖输出根目录。 |
| `--language` | 覆盖输出语言，可选 `zh` 或 `en`。 |

## 输出目录

主研报默认输出到：

```text
{output_dir}/{target_name}/
```

研报复现默认输出到：

```text
{output_dir}/{target_name}/report_reproduction/{report_id}/
```

研报复现会生成：

```text
parsed_report.md
strategy_brief.json
sample_strategy.py
README_strategy.md
manifest.json
```

## 常见问题

- `config validate` 提示环境变量未解析：检查 `.env` 或当前 shell 是否设置了对应变量。
- PDF 复现提示未提取到正文：当前轻量解析器主要支持文本型 PDF，扫描版 PDF 需要后续接入 OCR/MinerU 后端。
- 主研报运行中断：再次执行 `QAssistant report run --config my_config.yaml --resume` 可尝试从 checkpoint 继续。
