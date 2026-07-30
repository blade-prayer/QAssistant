import time

import pytest

from src.utils.rate_limiter import RateLimiter


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_unconfigured_service_no_delay(self):
        limiter = RateLimiter({"other": 5.0})
        start = time.monotonic()
        await limiter.acquire("unknown_service")
        assert time.monotonic() - start < 0.1

    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self):
        limiter = RateLimiter({"api": 0.3})
        await limiter.acquire("api")
        start = time.monotonic()
        await limiter.acquire("api")
        assert time.monotonic() - start >= 0.25

    @pytest.mark.asyncio
    async def test_per_service_isolation(self):
        limiter = RateLimiter({"slow": 1.0, "fast": 0.0})
        await limiter.acquire("slow")
        start = time.monotonic()
        await limiter.acquire("fast")
        assert time.monotonic() - start < 0.1

    @pytest.mark.asyncio
    async def test_set_interval_runtime(self):
        limiter = RateLimiter({})
        start = time.monotonic()
        await limiter.acquire("svc")
        await limiter.acquire("svc")
        assert time.monotonic() - start < 0.1

        limiter.set_interval("svc", 0.3)
        await limiter.acquire("svc")
        start = time.monotonic()
        await limiter.acquire("svc")
        assert time.monotonic() - start >= 0.25
