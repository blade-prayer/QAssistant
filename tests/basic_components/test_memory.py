import asyncio
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

root = str(Path(__file__).resolve().parents[2])
sys.path.append(root)

from src.memory import Memory
from src.memory.variable_memory import Memory as CompatMemory


class DummyLLM:
    async def generate(self, messages, **kwargs):
        return '["collect task from llm", "second collect task"]'


class DummyRecord:
    def __init__(self, name, description, data, source):
        self.name = name
        self.description = description
        self.data = data
        self.source = source

    def brief_str(self):
        return f"{self.name}: {self.description}"


class SearchResult(DummyRecord):
    def __init__(self, name, description, data, source, link):
        super().__init__(name, description, data, source)
        self.link = link


class ClickResult(DummyRecord):
    def __init__(self, name, description, data, source, link):
        super().__init__(name, description, data, source)
        self.link = link


class AnalysisResult:
    def __init__(self, title, content):
        self.title = title
        self.content = content

    def brief_str(self):
        return self.content

    def __str__(self):
        return f"{self.title}: {self.content}"


class DummyAgent:
    AGENT_NAME = "dummy_agent"

    def __init__(self, config, memory, agent_id=None, **kwargs):
        self.config = config
        self.memory = memory
        self.id = agent_id or "agent_dummy_agent_00000000"
        self.kwargs = kwargs

    @classmethod
    async def from_checkpoint(cls, config, memory, agent_id, **kwargs):
        return cls(config=config, memory=memory, agent_id=agent_id, **kwargs)


async def main():
    with TemporaryDirectory() as tmp_dir:
        config = SimpleNamespace(
            working_dir=tmp_dir,
            config={
                "target_name": "Smoke Target",
                "target_type": "financial_company",
            },
            llm_dict={"dummy-llm": DummyLLM()},
        )

        memory = Memory(config=config)
        assert CompatMemory is Memory
        record = DummyRecord("metric", "demo data", {"value": 1}, "unit-test")
        memory.add_data(record)
        memory.add_data(record)
        assert len(memory.get_collect_data()) == 1
        assert len(memory.get_records()) == 1
        first_record = memory.get_records()[0]
        assert first_record.memory_type == "collect"
        assert first_record.semantic_key
        assert first_record.content_hash
        assert first_record.quality_score > 0
        assert memory.get_record(first_record.id) is first_record
        assert memory.get_record_by_semantic_key(first_record.semantic_key) is first_record

        search_record = SearchResult("search title", "search desc", "summary", "search", "https://example.com")
        click_record = ClickResult("page", "page desc", "content", "URL: https://example.com", "https://example.com")
        analysis_record = AnalysisResult("analysis title", "analysis content")
        memory.add_data(search_record)
        memory.add_data(click_record)
        memory.add_data(analysis_record)
        url_record = memory.get_record_by_semantic_key("url:https://example.com")
        assert url_record is not None
        assert url_record.memory_type == "document"
        assert url_record.metadata.get("search_title") == "search title"
        assert url_record.metadata.get("versions")
        assert len(memory.get_records(memory_type="click")) == 1
        assert len(memory.get_collect_data(exclude_type=["search", "click"])) == 1
        assert memory.get_url_title("https://example.com") == "search title"
        assert len(memory.get_analysis_result()) == 1

        memory.add_dependency("child_tool", "parent_agent")
        memory.add_log("parent_agent", "agent_dummy", {"input": 1}, {"output": 2})
        assert "child_tool" in memory.dependency["parent_agent"]
        assert len(memory.log) == 1
        memory.add_log("child_tool", "tool_dummy", {"input": 2}, {"output": 3})
        assert memory.get_log("parent_agent")
        assert memory.get_log_by_type("tool_dummy")
        assert "metric" in memory.get_formatted_data_description([record])
        assert "analysis title" in memory.get_formatted_analysis_result()

        collect_tasks = await memory.generate_collect_tasks(
            query="Smoke Target",
            use_llm_name="dummy-llm",
            max_num=2,
        )
        assert collect_tasks
        assert memory.generated_collect_tasks

        task_input = {"input_data": {"task": "smoke task"}}
        agent = await memory.get_or_create_agent(
            agent_class=DummyAgent,
            task_input=task_input,
            resume=False,
            priority=1,
        )
        restored = await memory.get_or_create_agent(
            agent_class=DummyAgent,
            task_input=task_input,
            resume=True,
            priority=1,
        )
        assert restored.id == agent.id
        assert len(memory.task_mapping) == 1

        cache_dir = Path(tmp_dir) / "agent_working" / agent.id / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / "latest.pkl", "wb") as f:
            pickle.dump({"finished": True}, f)
        assert memory.is_agent_finished(agent.id)

        memory.save()
        loaded = Memory(config=config)
        state = loaded.load()
        assert state is not None
        assert loaded.metadata["schema_version"] == 2
        assert loaded.records
        assert len(loaded.get_collect_data(exclude_type=["search", "click"])) == 1
        assert loaded.get_url_title("https://example.com") == "search title"
        assert loaded.get_records(memory_type="click")
        assert len(loaded.task_mapping) == 1
        assert loaded.generated_collect_tasks

    with TemporaryDirectory() as tmp_dir:
        config = SimpleNamespace(
            working_dir=tmp_dir,
            config={
                "target_name": "Legacy Target",
                "target_type": "financial_company",
            },
            llm_dict={},
        )
        legacy_memory = Memory(config=config)
        legacy_record = DummyRecord("legacy metric", "legacy data", {"value": 2}, "legacy-source")
        legacy_snapshot = {
            "schema_version": 1,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "data": [legacy_record, legacy_record],
            "data_signatures": [],
            "task_mapping": [],
            "dependency": {},
            "log": [],
            "generated_collect_tasks": [],
            "generated_analysis_tasks": [],
        }
        with open(legacy_memory.memory_file, "wb") as f:
            pickle.dump(legacy_snapshot, f)

        migrated = Memory(config=config)
        state = migrated.load()
        assert state is not None
        assert migrated.metadata["schema_version"] == 2
        assert len(migrated.get_collect_data()) == 1
        migrated_record = migrated.get_records()[0]
        assert migrated_record.memory_type == "collect"
        assert migrated.get_record_by_semantic_key(migrated_record.semantic_key) is migrated_record

        print("Memory smoke test passed")


if __name__ == "__main__":
    asyncio.run(main())
