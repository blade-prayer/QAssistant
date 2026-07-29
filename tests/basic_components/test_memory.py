import asyncio
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

root = str(Path(__file__).resolve().parents[2])
sys.path.append(root)

from src.memory import Memory


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
        record = DummyRecord("metric", "demo data", {"value": 1}, "unit-test")
        memory.add_data(record)
        memory.add_data(record)
        assert len(memory.get_collect_data()) == 1

        memory.add_dependency("child_tool", "parent_agent")
        memory.add_log("parent_agent", "agent_dummy", {"input": 1}, {"output": 2})
        assert "child_tool" in memory.dependency["parent_agent"]
        assert len(memory.log) == 1

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
        assert len(loaded.get_collect_data()) == 1
        assert len(loaded.task_mapping) == 1
        assert loaded.generated_collect_tasks

        print("Memory smoke test passed")


if __name__ == "__main__":
    asyncio.run(main())
