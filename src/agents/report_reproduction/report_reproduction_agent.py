"""Lightweight report-reproduction agent.

This agent reads a local text-based PDF report, asks the LLM to extract a
structured strategy brief, then asks for auditable sample strategy code and a
README. It intentionally does not fetch market data or run a backtest.
"""

from __future__ import annotations

import ast
import json
import os
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.agents.base_agent import BaseAgent
from src.tools.base import ToolResult
from src.tools.document.pdf_report_parser import LocalPDFReportParser
from src.utils.prompt_loader import get_prompt_loader


BRIEF_SCHEMA_KEYS = [
    "strategy_name",
    "research_goal",
    "asset_universe",
    "factor_definitions",
    "entry_rules",
    "exit_rules",
    "rebalance_frequency",
    "risk_controls",
    "required_input_fields",
    "confirmed_from_pdf",
    "inferred_from_context",
    "implementation_assumptions",
    "missing_information",
    "page_evidence",
]


class ReportReproductionAgent(BaseAgent):
    """Generate sample strategy code from a local research-report PDF."""

    AGENT_NAME = "report_reproduction"
    AGENT_DESCRIPTION = (
        "Parse a local text-based PDF research report and generate a structured "
        "strategy brief, sample_strategy.py, and README_strategy.md. This MVP does "
        "not download data, run backtests, or validate factor performance."
    )
    NECESSARY_KEYS = ["pdf_path"]

    def __init__(
        self,
        config,
        tools: Optional[list] = None,
        use_llm_name: str = "deepseek-chat",
        enable_code: bool = False,
        memory=None,
        agent_id: Optional[str] = None,
    ):
        super().__init__(
            config=config,
            tools=[] if tools is None else tools,
            use_llm_name=use_llm_name,
            enable_code=enable_code,
            memory=memory,
            agent_id=agent_id,
        )
        self.prompt_loader = get_prompt_loader("report_reproduction", report_type="general")
        self.STRATEGY_BRIEF_PROMPT = self.prompt_loader.get_prompt("strategy_brief")
        self.STRATEGY_CODE_PROMPT = self.prompt_loader.get_prompt("strategy_code")
        self.pdf_parser = LocalPDFReportParser()

    def _set_default_tools(self):
        self.tools = []
        return self.tools

    async def _prepare_init_prompt(self, input_data: dict) -> list[dict]:
        pdf_path = input_data.get("pdf_path", "")
        return [{"role": "user", "content": f"Reproduce strategy from local PDF: {pdf_path}"}]

    def _target_language_name(self) -> str:
        target_language = getattr(self.config, "config", {}).get("language", "zh")
        language_mapping = {
            "zh": "Chinese (中文)",
            "en": "English",
        }
        return language_mapping.get(target_language, str(target_language))

    def _slugify(self, value: Any) -> str:
        text = str(value or "").strip()
        text = re.sub(r"[^\w.-]+", "_", text, flags=re.UNICODE).strip("._-")
        return text[:80] or "report"

    def _resolve_report_id(self, input_data: Dict[str, Any], pdf_path: str) -> str:
        if input_data.get("report_id"):
            return self._slugify(input_data["report_id"])
        return self._slugify(Path(pdf_path).stem)

    def _resolve_output_dir(self, input_data: Dict[str, Any], report_id: str) -> Path:
        base_dir = input_data.get("output_dir") or getattr(self.config, "working_dir", "./outputs")
        return Path(base_dir).expanduser() / "report_reproduction" / report_id

    def _write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _trim_text(self, text: str, max_chars: int = 60000) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[TRUNCATED: parsed PDF text exceeded prompt budget.]\n"

    def _json_loads(self, text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            pass
        try:
            import json_repair  # type: ignore

            return json_repair.loads(text)
        except Exception:
            return None

    def _parse_json_response(self, response: Any) -> Dict[str, Any]:
        text = str(response or "").strip()
        if not text:
            return {}

        parsed = self._json_loads(text)
        if isinstance(parsed, dict):
            return parsed

        fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        for block in reversed(fenced_blocks):
            parsed = self._json_loads(block.strip())
            if isinstance(parsed, dict):
                return parsed

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = self._json_loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        return {}

    def _listify(self, value: Any) -> List[Any]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _normalize_strategy_brief(self, raw: Dict[str, Any], parsed_report: Dict[str, Any]) -> Dict[str, Any]:
        brief: Dict[str, Any] = {}
        for key in BRIEF_SCHEMA_KEYS:
            brief[key] = raw.get(key)

        brief["strategy_name"] = str(brief.get("strategy_name") or "未命名研报复现策略")
        brief["research_goal"] = str(brief.get("research_goal") or "从 PDF 研报中复现策略规则并生成样例代码")
        brief["asset_universe"] = self._listify(brief.get("asset_universe"))
        brief["factor_definitions"] = self._listify(brief.get("factor_definitions"))
        brief["entry_rules"] = self._listify(brief.get("entry_rules"))
        brief["exit_rules"] = self._listify(brief.get("exit_rules"))
        brief["risk_controls"] = self._listify(brief.get("risk_controls"))
        brief["required_input_fields"] = self._listify(brief.get("required_input_fields"))
        brief["confirmed_from_pdf"] = self._listify(brief.get("confirmed_from_pdf"))
        brief["inferred_from_context"] = self._listify(brief.get("inferred_from_context"))
        brief["implementation_assumptions"] = self._listify(brief.get("implementation_assumptions"))
        brief["missing_information"] = self._listify(brief.get("missing_information"))
        brief["page_evidence"] = self._listify(brief.get("page_evidence"))
        brief["rebalance_frequency"] = str(brief.get("rebalance_frequency") or "未在 PDF 中明确")

        if not brief["required_input_fields"]:
            brief["required_input_fields"] = ["date", "asset"]
            brief["implementation_assumptions"].append(
                "PDF 未给出完整输入字段，样例代码默认至少需要 date 与 asset 字段。"
            )

        if not any((page.get("text") or "").strip() for page in parsed_report.get("pages", []) or []):
            brief["missing_information"].append("PDF 未提取到可用正文文本，可能需要 OCR 后再复现。")

        return brief

    def _extract_fenced_block(self, text: str, language: str) -> str:
        pattern = rf"```(?:{re.escape(language)})?\s*([\s\S]*?)```"
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        return matches[-1].strip() if matches else ""

    def _parse_code_response(
        self,
        response: Any,
        brief: Dict[str, Any],
        parsed_report: Dict[str, Any],
    ) -> Tuple[str, str]:
        text = str(response or "").strip()
        parsed = self._parse_json_response(text)
        code = ""
        readme = ""
        if parsed:
            code = str(
                parsed.get("sample_strategy_py")
                or parsed.get("sample_strategy.py")
                or parsed.get("code")
                or ""
            ).strip()
            readme = str(
                parsed.get("readme_strategy_md")
                or parsed.get("README_strategy.md")
                or parsed.get("readme")
                or ""
            ).strip()

        if not code:
            code = self._extract_fenced_block(text, "python")
        if not readme:
            readme = self._extract_fenced_block(text, "markdown")
        if not code:
            code = self._fallback_strategy_code(brief)
        if not readme:
            readme = self._build_readme(brief, parsed_report)
        return code.rstrip() + "\n", readme.rstrip() + "\n"

    def _page_evidence_comment(self, brief: Dict[str, Any]) -> str:
        evidence = brief.get("page_evidence") or []
        fragments = []
        for item in evidence[:5]:
            if isinstance(item, dict):
                page = item.get("page") or item.get("page_no") or item.get("pages")
                claim = item.get("claim") or item.get("evidence") or item.get("text")
                fragments.append(f"Page {page}: {claim}" if page else str(claim or item))
            else:
                fragments.append(str(item))
        return "; ".join(fragment for fragment in fragments if fragment) or "页码证据未明确"

    def _fallback_strategy_code(self, brief: Dict[str, Any]) -> str:
        required_fields = [str(item) for item in brief.get("required_input_fields", []) if str(item).strip()]
        if "date" not in required_fields:
            required_fields.insert(0, "date")
        if "asset" not in required_fields:
            required_fields.insert(1, "asset")
        factor_columns = [
            field for field in required_fields
            if field not in {"date", "asset", "open", "high", "low", "close", "volume"}
        ]
        page_evidence = self._page_evidence_comment(brief)
        params = {
            "strategy_name": brief.get("strategy_name", "report_reproduction_strategy"),
            "date_col": "date",
            "asset_col": "asset",
            "required_fields": required_fields,
            "factor_columns": factor_columns,
            "top_n": 10,
            "max_weight": 0.10,
            "rebalance_frequency": brief.get("rebalance_frequency", "未在 PDF 中明确"),
        }
        params_literal = json.dumps(params, ensure_ascii=False, indent=4)
        return textwrap.dedent(
            f'''\
            """研报复现策略样例代码。

            该文件由 FinSight 研报复现 MVP 自动生成，目标是提供可读、可改、
            结构清晰的策略骨架。它不会联网拉取数据，也不会运行真实回测。
            """

            from __future__ import annotations

            import pandas as pd


            PARAMS = {params_literal}


            def load_data_placeholder():
                """加载策略输入数据的占位函数。

                第一版 MVP 不接入数据源。请在二次开发时将本函数替换为本地 CSV、
                数据库或研究平台中的行情、因子与成分股数据读取逻辑。
                """
                raise NotImplementedError("请替换为本地数据读取逻辑；本样例不联网、不拉数。")


            def validate_input_schema(df: pd.DataFrame) -> pd.DataFrame:
                """检查输入字段是否满足策略样例运行要求。"""
                missing = [field for field in PARAMS["required_fields"] if field not in df.columns]
                if missing:
                    raise ValueError(f"输入数据缺少必要字段: {{missing}}")
                return df.copy()


            def compute_factors(df: pd.DataFrame, params: dict) -> pd.DataFrame:
                """计算策略因子。

                PDF 页码证据: {page_evidence}
                实现假设: 若 PDF 未给出精确公式，则对可用因子字段做截面百分位排名。
                """
                result = df.copy()
                factor_columns = [col for col in params.get("factor_columns", []) if col in result.columns]
                rank_columns = []
                if factor_columns:
                    for col in factor_columns:
                        rank_col = f"{{col}}_rank"
                        result[rank_col] = result.groupby(params["date_col"])[col].rank(pct=True)
                        rank_columns.append(rank_col)
                    result["factor_score"] = result[rank_columns].mean(axis=1)
                else:
                    result["factor_score"] = 0.0
                return result


            def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
                """根据因子得分生成持仓信号。

                实现假设: 每个调仓截面选择 factor_score 最高的 top_n 个资产。
                """
                result = df.copy()
                result["signal"] = 0
                top_n = int(params.get("top_n", 10))
                for _, group in result.groupby(params["date_col"], sort=False):
                    selected_index = group.nlargest(top_n, "factor_score").index
                    result.loc[selected_index, "signal"] = 1
                return result


            def build_target_weights(signals: pd.DataFrame, params: dict) -> pd.DataFrame:
                """将信号转换为目标权重。

                实现假设: 入选资产等权配置，并使用 max_weight 作为单资产权重上限。
                """
                result = signals.copy()
                result["target_weight"] = 0.0
                max_weight = float(params.get("max_weight", 0.10))
                for _, group in result.groupby(params["date_col"], sort=False):
                    selected = group[group["signal"] == 1]
                    if selected.empty:
                        continue
                    weight = min(1.0 / len(selected), max_weight)
                    result.loc[selected.index, "target_weight"] = weight
                return result


            def run_strategy_example(df: pd.DataFrame, params: dict = PARAMS) -> pd.DataFrame:
                """运行策略样例流程并返回带有信号和目标权重的数据。"""
                checked = validate_input_schema(df)
                factors = compute_factors(checked, params)
                signals = generate_signals(factors, params)
                weights = build_target_weights(signals, params)
                return weights


            def main():
                """命令行示例入口。

                本样例不会自动读取外部数据。请先准备 DataFrame，再调用
                run_strategy_example(df, PARAMS)。
                """
                print("请在二次开发时接入本地数据后调用 run_strategy_example(df, PARAMS)。")


            if __name__ == "__main__":
                main()
            '''
        ).strip() + "\n"

    def _build_readme(self, brief: Dict[str, Any], parsed_report: Dict[str, Any]) -> str:
        required_fields = "\n".join(f"- {item}" for item in brief.get("required_input_fields", [])) or "- 未明确"
        missing = "\n".join(f"- {item}" for item in brief.get("missing_information", [])) or "- 未发现"
        assumptions = "\n".join(f"- {item}" for item in brief.get("implementation_assumptions", [])) or "- 未发现"
        confirmed = "\n".join(f"- {item}" for item in brief.get("confirmed_from_pdf", [])) or "- 未明确"
        warnings = "\n".join(f"- {item}" for item in parsed_report.get("warnings", [])) or "- 无"
        return textwrap.dedent(
            f"""\
            # {brief.get("strategy_name", "研报复现策略样例")}

            ## 产物说明
            `sample_strategy.py` 是根据本地 PDF 研报生成的策略样例代码，目标是便于人工阅读、修改和继续开发。它不联网、不拉取真实数据、不接入回测框架，也不生成收益指标。

            ## 输入字段要求
            {required_fields}

            ## PDF 已确认信息
            {confirmed}

            ## 实现假设
            {assumptions}

            ## 缺失信息
            {missing}

            ## 解析告警
            {warnings}

            ## 使用边界
            该样例只表达策略逻辑骨架，不构成投资建议，也不代表策略已通过历史回测或实盘验证。
            """
        ).strip() + "\n"

    def _code_has_required_structure(self, code: str) -> bool:
        required_markers = [
            "PARAMS",
            "def load_data_placeholder(",
            "def validate_input_schema(",
            "def compute_factors(",
            "def generate_signals(",
            "def build_target_weights(",
            "def run_strategy_example(",
            "def main(",
            'if __name__ == "__main__"',
        ]
        return all(marker in code for marker in required_markers)

    async def async_run(
        self,
        input_data: dict,
        max_iterations: int = 2,
        stop_words: Optional[list[str]] = None,
        echo: bool = False,
        resume: bool = False,
        checkpoint_name: str = "report_reproduction_latest.pkl",
        prompt_function=None,
    ) -> dict:
        input_data = input_data or {}
        self._check_necessary_data(input_data)
        self.current_task_data = input_data

        pdf_path = input_data.get("pdf_path")
        if not pdf_path:
            raise ValueError("Input data must contain a 'pdf_path' key.")

        report_id = self._resolve_report_id(input_data, pdf_path)
        output_dir = self._resolve_output_dir(input_data, report_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        max_pages = input_data.get("max_pages")
        parsed_report = self.pdf_parser.parse_pdf(pdf_path=pdf_path, max_pages=max_pages)
        parsed_report_markdown = self.pdf_parser.to_markdown(parsed_report)

        parsed_report_path = output_dir / "parsed_report.md"
        brief_path = output_dir / "strategy_brief.json"
        code_path = output_dir / "sample_strategy.py"
        readme_path = output_dir / "README_strategy.md"
        manifest_path = output_dir / "manifest.json"

        self._write_text(parsed_report_path, parsed_report_markdown)

        warnings = list(parsed_report.get("warnings") or [])
        brief_prompt = self.STRATEGY_BRIEF_PROMPT.format(
            source_pdf=parsed_report.get("source_pdf", pdf_path),
            parsed_report_markdown=self._trim_text(parsed_report_markdown),
            target_language=self._target_language_name(),
        )
        brief_response = await self.llm.generate(messages=[{"role": "user", "content": brief_prompt}])
        strategy_brief = self._normalize_strategy_brief(
            self._parse_json_response(brief_response),
            parsed_report,
        )
        self._write_json(brief_path, strategy_brief)

        code_prompt = self.STRATEGY_CODE_PROMPT.format(
            source_pdf=parsed_report.get("source_pdf", pdf_path),
            strategy_brief_json=json.dumps(strategy_brief, ensure_ascii=False, indent=2),
            parsed_report_excerpt=self._trim_text(parsed_report_markdown, max_chars=30000),
            target_language=self._target_language_name(),
        )
        code_response = await self.llm.generate(messages=[{"role": "user", "content": code_prompt}])
        sample_code, readme = self._parse_code_response(code_response, strategy_brief, parsed_report)

        try:
            ast.parse(sample_code)
            if not self._code_has_required_structure(sample_code):
                warnings.append("LLM generated code missed the required strategy skeleton; fallback template was used.")
                sample_code = self._fallback_strategy_code(strategy_brief)
                ast.parse(sample_code)
        except SyntaxError as exc:
            warnings.append(f"LLM generated code had invalid Python syntax; fallback template was used: {exc}")
            sample_code = self._fallback_strategy_code(strategy_brief)
            ast.parse(sample_code)

        self._write_text(code_path, sample_code)
        self._write_text(readme_path, readme)

        generated_at = datetime.now().isoformat(timespec="seconds")
        model_name = getattr(self.llm, "model_name", None) or self.use_llm_name
        manifest = {
            "report_id": report_id,
            "source_pdf": str(parsed_report.get("source_pdf", pdf_path)),
            "generated_at": generated_at,
            "model_name": str(model_name),
            "page_count": parsed_report.get("page_count", 0),
            "parsed_page_count": parsed_report.get("parsed_page_count", 0),
            "warnings": warnings,
            "artifacts": {
                "parsed_report_md": str(parsed_report_path),
                "strategy_brief_json": str(brief_path),
                "sample_strategy_py": str(code_path),
                "readme_strategy_md": str(readme_path),
                "manifest_json": str(manifest_path),
            },
        }
        self._write_json(manifest_path, manifest)

        result = ToolResult(
            name=f"Report reproduction artifacts: {report_id}",
            description=(
                "Generated parsed_report.md, strategy_brief.json, sample_strategy.py, "
                "README_strategy.md, and manifest.json from a local PDF."
            ),
            data=manifest,
            source=f"Local PDF: {parsed_report.get('source_pdf', pdf_path)}",
        )
        if self.memory is not None:
            self.memory.add_data(result, source_agent_id=self.id, task_id=str(pdf_path), tool_name=self.AGENT_NAME)
            self.memory.add_log(
                id=self.id,
                type=self.type,
                input_data=input_data,
                output_data=manifest,
                error=False,
                note="Report reproduction artifacts generated successfully",
            )
            try:
                self.memory.save()
            except Exception:
                pass

        return_dict = {
            "input_data": input_data,
            "working_dir": self.working_dir,
            "output_dir": str(output_dir),
            "report_id": report_id,
            "strategy_brief": strategy_brief,
            "manifest": manifest,
            "final_result": str(manifest_path),
        }
        await self.save(
            state={
                "input_data": input_data,
                "return_dict": return_dict,
                "finished": True,
            },
            checkpoint_name=checkpoint_name,
        )
        return return_dict
