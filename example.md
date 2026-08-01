# QAssistant CLI 使用示例

`QAssistant` 是本项目的命令行入口，用于运行多智能体金融研究工作流。

## 安装

```powershell
pip install -e .
```

## 环境变量示例

```powershell
$env:DS_MODEL_NAME="MY_DS_MODEL_NAME"
$env:DS_BASE_URL="MY_DS_BASE_URL"
$env:DS_API_KEY="MY_DS_API_KEY"
$env:VLM_MODEL_NAME="MY_VLM_MODEL_NAME"
$env:VLM_BASE_URL="MY_VLM_BASE_URL"
$env:VLM_API_KEY="MY_VLM_API_KEY"
$env:EMBEDDING_MODEL_NAME="MY_EMBEDDING_MODEL_NAME"
$env:EMBEDDING_BASE_URL="MY_EMBEDDING_BASE_URL"
$env:EMBEDDING_API_KEY="MY_EMBEDDING_API_KEY"
```

## 主研报生成

```powershell
QAssistant report run --config my_config.yaml --resume --max-concurrent 3
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--config` | 配置文件路径，默认 `my_config.yaml`。 |
| `--tasks-file` | 可选任务文件。 |
| `--resume` | 从已有 checkpoint 继续运行。 |
| `--max-concurrent` | 同一阶段最大并发 agent 数。 |
| `--no-auto-tasks` | 不额外生成任务，只使用配置或任务文件中的任务。 |
| `--json` | 输出 JSON 摘要。 |

## 研报复现

```powershell
QAssistant reproduce --pdf-path report.pdf --config my_config.yaml --report-id sample
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--pdf-path` | 本地 PDF 研报路径。 |
| `--report-id` | 复现产物目录名。 |
| `--max-pages` | 限制解析页数，适合快速试跑。 |
| `--model-name` | 覆盖 LLM 模型名。 |

## 配置校验

```powershell
QAssistant config validate --config my_config.yaml --strict-env
```

`--strict-env` 会把缺失环境变量视为错误；不加时会作为 warning 输出。

## 输出产物列表

```powershell
QAssistant outputs list --config my_config.yaml --json
```

该命令会扫描配置对应的输出目录，列出报告文件和研报复现产物。
