import asyncio
import threading

import pytest

from src.utils.async_bridge import AsyncBridge, get_async_bridge
from src.utils.run_context import RunContext, get_run_context, run_context_scope


async def _add(a: int, b: int) -> int:
    await asyncio.sleep(0.01)
    return a + b


async def _fail():
    raise ValueError("intentional failure")


async def _slow(seconds: float):
    await asyncio.sleep(seconds)
    return "done"


async def _read_context():
    return get_run_context().to_dict()


class TestAsyncBridge:
    def test_run_async_from_sync(self):
        bridge = AsyncBridge(timeout=10)
        try:
            assert bridge.run_async(_add(3, 4)) == 7
        finally:
            bridge.shutdown()

    def test_run_async_from_inside_event_loop(self):
        bridge = AsyncBridge(timeout=10)
        try:
            async def _inner():
                return bridge.run_async(_add(10, 20))

            assert asyncio.run(_inner()) == 30
        finally:
            bridge.shutdown()

    def test_exception_propagation(self):
        bridge = AsyncBridge(timeout=10)
        try:
            with pytest.raises(ValueError, match="intentional failure"):
                bridge.run_async(_fail())
        finally:
            bridge.shutdown()

    def test_timeout(self):
        bridge = AsyncBridge(timeout=0.2)
        try:
            with pytest.raises(TimeoutError):
                bridge.run_async(_slow(5.0))
        finally:
            bridge.shutdown()

    def test_multiple_concurrent_calls(self):
        bridge = AsyncBridge(timeout=10)
        results = [None] * 5
        errors = []

        def _worker(idx: int):
            try:
                results[idx] = bridge.run_async(_add(idx, idx))
            except Exception as exc:
                errors.append(exc)

        try:
            threads = [threading.Thread(target=_worker, args=(idx,)) for idx in range(5)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            assert not errors
            assert results == [0, 2, 4, 6, 8]
        finally:
            bridge.shutdown()

    def test_run_async_preserves_run_context(self):
        bridge = AsyncBridge(timeout=10)
        try:
            with run_context_scope(RunContext(
                run_id="run_ctx",
                agent_id="agent_ctx",
                agent_name="ctx_agent",
                task_id="ctx_task",
                step_id=7,
                tool_name="ctx_tool",
            )):
                result = bridge.run_async(_read_context())
            assert result["run_id"] == "run_ctx"
            assert result["agent_id"] == "agent_ctx"
            assert result["agent_name"] == "ctx_agent"
            assert result["task_id"] == "ctx_task"
            assert result["step_id"] == 7
            assert result["tool_name"] == "ctx_tool"
        finally:
            bridge.shutdown()


def test_get_async_bridge_singleton_works():
    bridge = get_async_bridge()
    assert bridge is get_async_bridge()
    assert bridge.run_async(_add(100, 200)) == 300
