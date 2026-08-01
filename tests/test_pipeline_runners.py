import asyncio
from pathlib import Path

from src.pipeline.report_runner import run_report_pipeline
from src.pipeline.reproduction_runner import run_reproduction_pipeline


class _FakeLLM:
    model_name = "fake-model"


def _make_config_cls(tmp_path):
    class FakeConfig:
        def __init__(self, config_file_path=None, config_dict=None):
            self.config = {
                "target_name": "DemoTarget",
                "stock_code": "000001",
                "target_type": "company",
                "output_dir": str(tmp_path / "outputs"),
                "language": "zh",
                "custom_collect_tasks": ["Collect revenue"],
                "custom_analysis_tasks": ["Analyze profitability"],
            }
            self.config.update(config_dict or {})
            self.working_dir = str(tmp_path / "outputs" / self.config["target_name"])
            self.llm_dict = {"fake-model": _FakeLLM()}

    return FakeConfig


def test_report_runner_builds_and_runs_three_phase_tasks(tmp_path):
    created = []
    runs = []

    class FakeMemory:
        def __init__(self, config):
            self.config = config
            self.metadata = {}
            self.task_mapping = []
            self.generated_collect_tasks = []
            self.generated_analysis_tasks = []

        async def get_or_create_agent(self, agent_class, task_input, resume, priority, **agent_kwargs):
            agent = agent_class(config=self.config, memory=self, **agent_kwargs)
            created.append({
                "agent": agent,
                "task_input": task_input,
                "resume": resume,
                "priority": priority,
                "agent_kwargs": agent_kwargs,
            })
            self.task_mapping.append({"agent_id": agent.id, "priority": priority})
            return agent

        def is_agent_finished(self, agent_id):
            return False

        def save(self):
            return None

    class FakeAgent:
        AGENT_NAME = "fake_agent"

        def __init__(self, config, memory, **kwargs):
            self.config = config
            self.memory = memory
            self.kwargs = kwargs
            self.id = f"{self.AGENT_NAME}_{len(created) + 1}"

        async def async_run(self, **kwargs):
            runs.append({"agent_id": self.id, "kwargs": kwargs})
            return {"final_result": "ok"}

    class FakeCollector(FakeAgent):
        AGENT_NAME = "collector"

    class FakeAnalyzer(FakeAgent):
        AGENT_NAME = "analyzer"

    class FakeGenerator(FakeAgent):
        AGENT_NAME = "generator"

    result = asyncio.run(run_report_pipeline(
        config_file_path="unused.yaml",
        resume=False,
        max_concurrent=2,
        use_llm_name="fake-model",
        use_vlm_name="fake-model",
        use_embedding_name="fake-model",
        auto_generate_tasks=False,
        config_cls=_make_config_cls(tmp_path),
        memory_cls=FakeMemory,
        agent_classes={
            "collector": FakeCollector,
            "analyzer": FakeAnalyzer,
            "generator": FakeGenerator,
        },
        runtime_helpers={},
    ))

    assert result["status"] == "success"
    assert result["collect_task_count"] == 1
    assert result["analysis_task_count"] == 1
    assert result["agent_count"] == 3
    assert len(created) == 3
    assert len(runs) == 3
    assert created[0]["priority"] == 1
    assert created[1]["priority"] == 2
    assert created[2]["priority"] == 3
    assert created[0]["agent_kwargs"]["use_llm_name"] == "fake-model"


def test_reproduction_runner_passes_cli_arguments_to_agent(tmp_path):
    calls = {}

    class FakeMemory:
        def __init__(self, config):
            self.config = config

    class FakeAgent:
        def __init__(self, config, memory, use_llm_name, enable_code):
            calls["init"] = {
                "config": config,
                "memory": memory,
                "use_llm_name": use_llm_name,
                "enable_code": enable_code,
            }

        async def async_run(self, input_data, resume):
            calls["run"] = {"input_data": input_data, "resume": resume}
            out = Path(calls["init"]["config"].working_dir) / "report_reproduction" / "sample"
            return {
                "report_id": "sample",
                "output_dir": str(out),
                "final_result": str(out / "manifest.json"),
                "manifest": {"warnings": ["parser warning"]},
            }

    result = asyncio.run(run_reproduction_pipeline(
        pdf_path="report.pdf",
        config_file_path="unused.yaml",
        config_overrides={"output_dir": str(tmp_path / "outputs")},
        report_id="sample",
        model_name="fake-model",
        max_pages=5,
        config_cls=_make_config_cls(tmp_path),
        memory_cls=FakeMemory,
        agent_cls=FakeAgent,
        runtime_helpers={},
    ))

    assert calls["init"]["use_llm_name"] == "fake-model"
    assert calls["init"]["enable_code"] is False
    assert calls["run"]["input_data"] == {
        "pdf_path": "report.pdf",
        "report_id": "sample",
        "max_pages": 5,
    }
    assert calls["run"]["resume"] is False
    assert result["summary"]["warnings"] == ["parser warning"]
