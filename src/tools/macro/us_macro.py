"""US macroeconomic and market data tools."""

from __future__ import annotations

import os

import pandas as pd

from ..base import Tool, ToolResult

_fred = None


def _get_fred():
    """Return a cached FRED client, or None if it is unavailable."""
    global _fred
    if _fred is not None:
        return _fred
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        return None
    try:
        from fredapi import Fred

        _fred = Fred(api_key=api_key)
        return _fred
    except Exception:
        return None


class USCPI(Tool):
    def __init__(self):
        super().__init__(
            name="US CPI (FRED)",
            description="US Consumer Price Index for All Urban Consumers (CPIAUCSL) from FRED. Monthly, seasonally adjusted.",
            parameters=[
                {"name": "start", "type": "str", "description": "Start date YYYY-MM-DD; default is 10 years ago", "required": False},
            ],
        )

    async def api_function(self, start: str = None):
        fred = _get_fred()
        if fred is None:
            return [ToolResult(self.name, "FRED_API_KEY not set", None, "FRED: CPIAUCSL")]
        try:
            if start is None:
                import datetime

                start = (datetime.date.today() - datetime.timedelta(days=365 * 10)).isoformat()
            series = fred.get_series("CPIAUCSL", observation_start=start)
            data = pd.DataFrame({"date": series.index, "CPI": series.values})
        except Exception as exc:
            print(f"Failed to fetch FRED CPI: {exc}")
            data = None
        return [ToolResult(self.name, "US CPI (All Urban, SA)", data, "FRED: CPIAUCSL https://fred.stlouisfed.org/series/CPIAUCSL")]


class USGDP(Tool):
    def __init__(self):
        super().__init__(
            name="US GDP (FRED)",
            description="US Gross Domestic Product (GDP) from FRED. Quarterly, seasonally adjusted annual rate.",
            parameters=[
                {"name": "start", "type": "str", "description": "Start date YYYY-MM-DD; default is 20 years ago", "required": False},
            ],
        )

    async def api_function(self, start: str = None):
        fred = _get_fred()
        if fred is None:
            return [ToolResult(self.name, "FRED_API_KEY not set", None, "FRED: GDP")]
        try:
            if start is None:
                import datetime

                start = (datetime.date.today() - datetime.timedelta(days=365 * 20)).isoformat()
            series = fred.get_series("GDP", observation_start=start)
            data = pd.DataFrame({"date": series.index, "GDP_billions": series.values})
        except Exception as exc:
            print(f"Failed to fetch FRED GDP: {exc}")
            data = None
        return [ToolResult(self.name, "US GDP (SAAR, billions USD)", data, "FRED: GDP https://fred.stlouisfed.org/series/GDP")]


class USUnemployment(Tool):
    def __init__(self):
        super().__init__(
            name="US Unemployment rate (FRED)",
            description="US Civilian Unemployment Rate (UNRATE) from FRED. Monthly, seasonally adjusted.",
            parameters=[
                {"name": "start", "type": "str", "description": "Start date YYYY-MM-DD; default is 10 years ago", "required": False},
            ],
        )

    async def api_function(self, start: str = None):
        fred = _get_fred()
        if fred is None:
            return [ToolResult(self.name, "FRED_API_KEY not set", None, "FRED: UNRATE")]
        try:
            if start is None:
                import datetime

                start = (datetime.date.today() - datetime.timedelta(days=365 * 10)).isoformat()
            series = fred.get_series("UNRATE", observation_start=start)
            data = pd.DataFrame({"date": series.index, "unemployment_rate_pct": series.values})
        except Exception as exc:
            print(f"Failed to fetch FRED Unemployment: {exc}")
            data = None
        return [ToolResult(self.name, "US Unemployment Rate (%)", data, "FRED: UNRATE https://fred.stlouisfed.org/series/UNRATE")]


class USInterestRates(Tool):
    def __init__(self):
        super().__init__(
            name="US Interest rates (FRED)",
            description="US key interest rates: Fed Funds Rate (FEDFUNDS), 10-Year Treasury (DGS10), 2-Year Treasury (DGS2).",
            parameters=[
                {"name": "start", "type": "str", "description": "Start date YYYY-MM-DD; default is 10 years ago", "required": False},
            ],
        )

    async def api_function(self, start: str = None):
        fred = _get_fred()
        if fred is None:
            return [ToolResult(self.name, "FRED_API_KEY not set", None, "FRED")]
        try:
            import datetime

            if start is None:
                start = (datetime.date.today() - datetime.timedelta(days=365 * 10)).isoformat()
            series_ids = {"FEDFUNDS": "fed_funds_rate", "DGS10": "treasury_10y", "DGS2": "treasury_2y"}
            frames = []
            for series_id, column_name in series_ids.items():
                series = fred.get_series(series_id, observation_start=start)
                frames.append(pd.DataFrame({"date": series.index, column_name: series.values}))
            data = frames[0]
            for frame in frames[1:]:
                data = data.merge(frame, on="date", how="outer")
            data = data.sort_values("date").reset_index(drop=True)
        except Exception as exc:
            print(f"Failed to fetch FRED interest rates: {exc}")
            data = None
        return [ToolResult(self.name, "US Key Interest Rates", data, "FRED: FEDFUNDS, DGS10, DGS2 https://fred.stlouisfed.org")]


class USMarketIndex(Tool):
    def __init__(self):
        super().__init__(
            name="US Market index",
            description="US major market index OHLCV data: S&P 500 (^GSPC), DJIA (^DJI), or NASDAQ (^IXIC) via Yahoo Finance.",
            parameters=[
                {"name": "index_symbol", "type": "str", "description": "Yahoo Finance index symbol: ^GSPC, ^DJI, or ^IXIC", "required": True},
                {"name": "period", "type": "str", "description": "Time period: 1mo, 3mo, 6mo, 1y, 2y, 5y, max", "required": False},
            ],
        )

    async def api_function(self, index_symbol: str = "^GSPC", period: str = "1y"):
        try:
            import yfinance as yf

            ticker = yf.Ticker(index_symbol)
            history = ticker.history(period=period)
            if history.empty:
                data = None
            else:
                history = history.reset_index()
                keep_cols = [column for column in ["Date", "Open", "High", "Low", "Close", "Volume"] if column in history.columns]
                data = history[keep_cols].copy()
                for column in ["Open", "High", "Low", "Close"]:
                    if column in data.columns:
                        data[column] = data[column].round(2)
        except Exception as exc:
            print(f"Failed to fetch US market index {index_symbol}: {exc}")
            data = None

        index_names = {"^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "NASDAQ Composite"}
        display_name = index_names.get(index_symbol, index_symbol)
        return [ToolResult(
            name=f"{display_name} ({period})",
            description=f"{display_name} index OHLCV data for {period}.",
            data=data,
            source=f"Yahoo Finance: https://finance.yahoo.com/quote/{index_symbol}",
        )]
