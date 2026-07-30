import pytest


def test_tool_registry_imports_and_lists_tools():
    from src.tools import list_tools

    tools = list_tools()
    assert isinstance(tools, list)
    assert "US CPI (FRED)" in tools
    assert "US Market index" in tools


def test_core_financial_tools_when_optional_deps_available():
    try:
        import akshare  # noqa: F401
        import efinance  # noqa: F401
    except Exception:
        pytest.skip("Chinese market optional dependencies are not installed")

    from src.tools import list_tools

    tools = list_tools()
    expected = [
        "Stock profile",
        "Stock candlestick data",
        "Balance sheet",
        "Income statement",
        "Cash-flow statement",
        "Shareholding structure",
    ]
    for name in expected:
        assert name in tools


def test_get_us_macro_tool_by_name():
    from src.tools import get_tool_by_name

    cls = get_tool_by_name("US CPI (FRED)")
    assert cls is not None
    assert cls().name == "US CPI (FRED)"
