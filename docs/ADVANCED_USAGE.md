# QAssistant Advanced Usage

本文档说明 QAssistant 的高级配置、CLI 运行方式和扩展点。项目当前不包含前后端展示功能，推荐通过 `QAssistant` 命令行入口运行。

## CLI 工作流

主研报生成：

```powershell
QAssistant report run --config my_config.yaml --resume --max-concurrent 3
```

研报复现：

```powershell
QAssistant reproduce --pdf-path report.pdf --config my_config.yaml --report-id sample
```

配置校验：

```powershell
QAssistant config validate --config my_config.yaml --strict-env
```

输出产物列表：

```powershell
QAssistant outputs list --config my_config.yaml --json
```

## 配置层次

QAssistant 使用两层配置：

- YAML/JSON 配置文件，例如 `my_config.yaml`。
- 环境变量，例如 `DS_API_KEY`、`VLM_API_KEY`、`EMBEDDING_API_KEY`。

示例环境变量值统一使用占位符：

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
```

配置文件应引用环境变量，而不是直接写入真实凭据：

```yaml
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

## Pipeline Runner

CLI 入口调用 `src.pipeline` 中的 runner：

- `run_report_pipeline(...)`：执行数据收集、数据分析、报告生成。
- `run_reproduction_pipeline(...)`：执行本地 PDF 解析、策略摘要生成、样例策略代码生成。

旧脚本 `run_report.py` 和 `run_report_reproduction.py` 保留为兼容包装器。

## 任务文件

`QAssistant report run --tasks-file tasks.json` 支持两种格式。

简化格式：

```json
{
  "custom_collect_tasks": ["收集公司财务报表"],
  "custom_analysis_tasks": ["分析盈利能力"]
}
```

兼容列表格式：

```json
{
  "collect_tasks": [{"content": "收集公司财务报表"}],
  "analysis_tasks": [{"content": "分析盈利能力"}]
}
```

## 扩展工具

工具位于 `src/tools/`，会通过现有自动注册机制发现。新增工具时建议：

- 继承基础 `Tool` 类。
- 返回 `ToolResult`。
- 将可选依赖做成懒加载，避免影响 CLI 启动。
- 为失败路径返回清晰 warning 或错误信息。

## 研报复现边界

PDF 复现功能生成的是策略样例代码，不承诺可交易或可回测。缺失参数应写入 `missing_information` 和 `implementation_assumptions`，不能默认为事实。
