import pytest

from src.tools.macro.us_macro import USCPI


@pytest.mark.asyncio
async def test_fred_tool_degrades_without_api_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    result = await USCPI().api_function()

    assert len(result) == 1
    assert result[0].data is None
    assert "FRED_API_KEY not set" in result[0].description
