"""Thread-safe bridge for calling async functions from sync code.

LLM-generated code runs inside ``exec()`` in a synchronous frame, while the
pipeline itself is already inside an asyncio event loop. Calling
``asyncio.run()`` from there raises ``RuntimeError`` or can deadlock. This
module owns a background event loop and lets synchronous helper functions wait
for coroutines safely.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import threading
from typing import Any, Coroutine, Optional


class AsyncBridge:
    """Run coroutines from synchronous code without nesting event loops."""

    def __init__(self, timeout: float = 300.0):
        self._timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="async-bridge-worker",
        )
        self._thread.start()

    def run_async(self, coro: Coroutine, context: Optional[contextvars.Context] = None) -> Any:
        ctx = context or contextvars.copy_context()
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()

        def _start() -> None:
            if future.cancelled():
                return
            try:
                task = ctx.run(self._loop.create_task, coro)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
                return

            def _done(task: asyncio.Task) -> None:
                if future.cancelled():
                    return
                try:
                    result = task.result()
                except Exception as exc:  # pragma: no cover - task error path
                    if not future.done():
                        future.set_exception(exc)
                    return
                if not future.done():
                    future.set_result(result)

            task.add_done_callback(_done)

        self._loop.call_soon_threadsafe(_start)
        return future.result(timeout=self._timeout)

    def shutdown(self) -> None:
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if not self._loop.is_closed():
            self._loop.close()


_bridge: AsyncBridge | None = None
_lock = threading.Lock()


def get_async_bridge(timeout: float = 300.0) -> AsyncBridge:
    """Return the shared bridge instance."""
    global _bridge
    if _bridge is None:
        with _lock:
            if _bridge is None:
                _bridge = AsyncBridge(timeout=timeout)
    return _bridge
