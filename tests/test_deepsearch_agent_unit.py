import asyncio
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
    json_repair.loads = lambda text: {}
    sys.modules.setdefault("json_repair", json_repair)


_install_optional_dependency_stubs()

from src.agents.search_agent.search_agent import DeepSearchAgent  # noqa: E402
from src.tools.web.base_search import SearchResult  # noqa: E402


class FakeLLM:
    def __init__(self):
        self.last_messages = None
        self.last_params = None

    async def generate(self, messages, **params):
        self.last_messages = messages
        self.last_params = params
        return "## Answer\nEvidence-backed fallback. [Source: Official Filing]\n\n## References\nOfficial Filing - https://example.com/filing"


class FakeConfig:
    def __init__(self, tmp_path):
        self.config = {"language": "en"}
        self.working_dir = str(tmp_path)
        self.llm_dict = {"fake-llm": FakeLLM()}


class FakeMemory:
    def __init__(self):
        self.data = []
        self.logs = []
        self.dependencies = []

    def add_dependency(self, child_id, parent_id):
        self.dependencies.append((child_id, parent_id))

    def add_data(self, item, **kwargs):
        self.data.append(item)
        return item

    def add_log(self, *args, **kwargs):
        self.logs.append({"args": args, "kwargs": kwargs})
        return self.logs[-1]

    def save(self):
        return None


class FakeSearchTool:
    name = "Fake Search"
    type = "tool_search"
    id = "tool_fake_search"

    def __init__(self, results):
        self.results = results
        self.calls = []

    async def api_function(self, query):
        self.calls.append(query)
        return self.results


class FakeClickResult:
    def __init__(self, data, link="https://example.com/filing", name="Fetched page"):
        self.data = data
        self.link = link
        self.name = name
        self.description = "Fetched page"
        self.source = f"URL: {link}"


class FakeClickTool:
    name = "Web page content fetcher"
    type = "tool_click"
    id = "tool_fake_click"

    def __init__(self, results):
        self.results = results
        self.calls = []

    async def api_function(self, urls, task):
        self.calls.append((urls, task))
        return self.results


def _make_agent(tmp_path, search_results=None, click_results=None):
    memory = FakeMemory()
    search_tool = FakeSearchTool(search_results or [])
    click_tool = FakeClickTool(click_results or [])
    agent = DeepSearchAgent(
        config=FakeConfig(tmp_path),
        tools=[search_tool, click_tool],
        use_llm_name="fake-llm",
        memory=memory,
    )
    agent.current_round = 0
    agent.max_iterations = 10
    agent.current_task_data = {"task": "Research target", "query": "revenue"}
    return agent, search_tool, click_tool, memory


def test_search_result_objects_populate_valid_links(tmp_path):
    search_results = [
        SearchResult(
            query="company filing revenue",
            name="Official Filing",
            description="Revenue and annual report details.",
            link="https://example.com/filing",
            data={"title": "Official Filing"},
            source="Official Filing",
        )
    ]
    agent, _, _, memory = _make_agent(tmp_path, search_results=search_results)

    result = asyncio.run(agent._handle_search_action("company filing revenue"))

    assert result["continue"] is True
    assert "https://example.com/filing" in agent.valid_links
    assert agent.valid_links["https://example.com/filing"]["title"] == "Official Filing"
    assert agent.link2name["https://example.com/filing"] == "Official Filing"
    assert memory.logs[-1]["kwargs"]["output_data"]["result"][0]["link"] == "https://example.com/filing"


def test_search_dicts_populate_valid_links(tmp_path):
    search_results = [
        {
            "title": "IR Presentation",
            "url": "https://example.com/ir",
            "snippet": "Investor relations update.",
        }
    ]
    agent, _, _, _ = _make_agent(tmp_path, search_results=search_results)

    asyncio.run(agent._handle_search_action("company investor relations"))

    assert agent.valid_links["https://example.com/ir"]["title"] == "IR Presentation"
    assert agent.valid_links["https://example.com/ir"]["description"] == "Investor relations update."


def test_click_registered_url_populates_used_sources(tmp_path):
    click_result = FakeClickResult("Detailed page content with useful facts.")
    agent, _, click_tool, memory = _make_agent(tmp_path, click_results=[click_result])
    agent.valid_links["https://example.com/filing"] = {
        "title": "Official Filing",
        "description": "Annual filing.",
        "query": "company filing",
    }
    agent.link2name["https://example.com/filing"] = "Official Filing"

    result = asyncio.run(agent._handle_click_action("https://example.com/filing"))

    assert result["result"] == "Detailed page content with useful facts."
    assert click_tool.calls[0][0] == ["https://example.com/filing"]
    assert agent.used_sources["https://example.com/filing"]["title"] == "Official Filing"
    assert memory.data[-1].name == "Official Filing"


def test_click_unregistered_url_is_rejected_without_calling_tool(tmp_path):
    agent, _, click_tool, _ = _make_agent(tmp_path, click_results=[FakeClickResult("content")])

    result = asyncio.run(agent._handle_click_action("https://example.com/not-from-search"))

    assert "was not found in your search results" in result["result"]
    assert click_tool.calls == []
    assert agent.used_sources == {}


def test_click_empty_crawler_result_does_not_raise(tmp_path):
    agent, _, click_tool, memory = _make_agent(tmp_path, click_results=[])
    agent.valid_links["https://example.com/empty"] = {
        "title": "Empty Page",
        "description": "No fetched content.",
        "query": "empty",
    }

    result = asyncio.run(agent._handle_click_action("https://example.com/empty"))

    assert "Failed to fetch content" in result["result"]
    assert click_tool.calls[0][0] == ["https://example.com/empty"]
    assert memory.logs[-1]["kwargs"]["error"] is True


def test_max_round_fallback_uses_plain_text_generation(tmp_path):
    agent, _, _, _ = _make_agent(tmp_path)
    agent.valid_links["https://example.com/filing"] = {
        "title": "Official Filing",
        "description": "Annual filing.",
        "query": "company filing",
    }

    result = asyncio.run(agent._handle_max_round([
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "<search>company filing</search>"},
    ]))

    llm = agent.config.llm_dict["fake-llm"]
    assert result["final_result"].startswith("## Answer")
    assert llm.last_params == {}
    assert "VERIFIED SOURCES AVAILABLE FOR CITATION" in llm.last_messages[0]["content"]
