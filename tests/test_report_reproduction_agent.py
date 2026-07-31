import ast
import asyncio
import json
import sys
import types
from pathlib import Path


root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.append(root)


def _install_optional_dependency_stubs():
    docx2pdf = types.ModuleType("docx2pdf")
    docx2pdf.convert = lambda *args, **kwargs: None
    sys.modules.setdefault("docx2pdf", docx2pdf)

    pdfplumber = types.ModuleType("pdfplumber")
    pdfplumber.open = lambda *args, **kwargs: None
    sys.modules.setdefault("pdfplumber", pdfplumber)

    class _FakeOpenAI:
        def __init__(self, *args, **kwargs):
            pass

    openai = types.ModuleType("openai")
    openai.OpenAI = _FakeOpenAI
    openai.AsyncOpenAI = _FakeOpenAI
    sys.modules.setdefault("openai", openai)

    json_repair = types.ModuleType("json_repair")
    json_repair.loads = lambda text: json.loads(text)
    sys.modules.setdefault("json_repair", json_repair)


_install_optional_dependency_stubs()

from src.agents.report_reproduction.report_reproduction_agent import ReportReproductionAgent  # noqa: E402


class FakePage:
    def extract_text(self):
        return "第1页：策略选择过去20日动量最高的股票，并按月调仓。"

    def extract_tables(self):
        return [[["字段", "含义"], ["momentum_20d", "过去20日收益率"]]]


class FakePDF:
    pages = [FakePage()]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _install_fake_pdfplumber(monkeypatch):
    fake_pdfplumber = types.ModuleType("pdfplumber")
    fake_pdfplumber.open = lambda path: FakePDF()
    monkeypatch.setitem(sys.modules, "pdfplumber", fake_pdfplumber)


SAMPLE_CODE = '''
from __future__ import annotations

import pandas as pd

PARAMS = {"required_fields": ["date", "asset", "momentum_20d"], "date_col": "date", "asset_col": "asset"}

def load_data_placeholder():
    """占位数据加载函数。"""
    raise NotImplementedError("本样例不联网、不拉数。")

def validate_input_schema(df):
    """检查必要字段。"""
    missing = [field for field in PARAMS["required_fields"] if field not in df.columns]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    return df.copy()

def compute_factors(df, params):
    """根据 PDF 第1页的动量描述计算因子。"""
    result = df.copy()
    result["factor_score"] = result["momentum_20d"]
    return result

def generate_signals(df, params):
    """生成持仓信号。"""
    result = df.copy()
    result["signal"] = result.groupby(params["date_col"])["factor_score"].rank(ascending=False) <= 10
    return result

def build_target_weights(signals, params):
    """生成目标权重。"""
    result = signals.copy()
    result["target_weight"] = result["signal"].astype(float)
    return result

def run_strategy_example(df, params=PARAMS):
    """运行样例策略。"""
    checked = validate_input_schema(df)
    factors = compute_factors(checked, params)
    signals = generate_signals(factors, params)
    return build_target_weights(signals, params)

def main():
    """命令行占位入口。"""
    print("sample strategy")

if __name__ == "__main__":
    main()
'''.strip()


class FakeLLM:
    model_name = "fake-model"

    def __init__(self):
        self.prompts = []

    async def generate(self, messages, **params):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            return json.dumps(
                {
                    "strategy_name": "动量轮动样例",
                    "research_goal": "复现 PDF 中的月度动量选股规则",
                    "asset_universe": ["A股股票"],
                    "factor_definitions": ["momentum_20d = 过去20日收益率"],
                    "entry_rules": ["每月选择 momentum_20d 最高的股票"],
                    "exit_rules": ["下次调仓未入选则退出"],
                    "rebalance_frequency": "monthly",
                    "risk_controls": ["等权"],
                    "required_input_fields": ["date", "asset", "momentum_20d"],
                    "confirmed_from_pdf": ["PDF 第1页提到20日动量和月度调仓"],
                    "inferred_from_context": ["样例代码使用截面排序"],
                    "implementation_assumptions": ["未给出交易成本，样例不计算费用"],
                    "missing_information": ["未给出股票池过滤细节"],
                    "page_evidence": [{"page": 1, "claim": "20日动量与月度调仓"}],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "sample_strategy_py": SAMPLE_CODE,
                "readme_strategy_md": "# 动量轮动样例\n\n本代码为样例，不联网、不回测。",
            },
            ensure_ascii=False,
        )


class FakeConfig:
    def __init__(self, tmp_path):
        self.config = {"language": "zh", "target_name": "测试标的"}
        self.working_dir = str(tmp_path)
        self.llm_dict = {"fake-model": FakeLLM()}


class FakeMemory:
    def __init__(self):
        self.data = []
        self.logs = []

    def add_dependency(self, child_id, parent_id):
        return None

    def add_data(self, item, **kwargs):
        self.data.append((item, kwargs))
        return item

    def add_log(self, *args, **kwargs):
        self.logs.append({"args": args, "kwargs": kwargs})
        return self.logs[-1]

    def save(self):
        return None


def test_report_reproduction_agent_generates_required_artifacts(monkeypatch, tmp_path):
    _install_fake_pdfplumber(monkeypatch)
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    config = FakeConfig(tmp_path)
    memory = FakeMemory()
    agent = ReportReproductionAgent(config=config, memory=memory, use_llm_name="fake-model")

    result = asyncio.run(agent.async_run({
        "pdf_path": str(pdf_path),
        "report_id": "demo_report",
    }))

    output_dir = Path(result["output_dir"])
    required_files = [
        "parsed_report.md",
        "strategy_brief.json",
        "sample_strategy.py",
        "README_strategy.md",
        "manifest.json",
    ]
    for file_name in required_files:
        assert (output_dir / file_name).exists()

    code = (output_dir / "sample_strategy.py").read_text(encoding="utf-8")
    ast.parse(code)
    for marker in [
        "PARAMS",
        "def load_data_placeholder(",
        "def validate_input_schema(",
        "def compute_factors(",
        "def generate_signals(",
        "def build_target_weights(",
        "def run_strategy_example(",
        "def main(",
        'if __name__ == "__main__"',
    ]:
        assert marker in code

    brief = json.loads((output_dir / "strategy_brief.json").read_text(encoding="utf-8"))
    assert brief["strategy_name"] == "动量轮动样例"
    assert "confirmed_from_pdf" in brief
    assert "implementation_assumptions" in brief

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["sample_strategy_py"].endswith("sample_strategy.py")
    assert len(memory.data) == 1
    assert len(memory.logs) == 1

    llm = config.llm_dict["fake-model"]
    assert "confirmed_from_pdf" in llm.prompts[0]
    assert "sample_strategy.py" in llm.prompts[1]
