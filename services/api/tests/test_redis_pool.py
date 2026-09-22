import asyncio
from unittest.mock import patch

import app.redis_pool as redis_pool_module
from app.redis_pool import get_redis_pool


def test_concurrent_callers_share_one_pool_instance(monkeypatch):
    # Regression test for a real bug: the lazy singleton's check-and-create
    # (`if _pool is None: _pool = await create_pool(...)`) wasn't
    # synchronized, so two requests arriving before the pool existed could
    # both pass the None check, both call create_pool(), and whichever
    # finished last would silently overwrite the other's pool — leaking
    # the loser's connection pool. A lock with a re-check inside it closes
    # that window.
    monkeypatch.setattr(redis_pool_module, "_pool", None)

    created_pools = []

    async def slow_create_pool(*args, **kwargs):
        # Forces every concurrent caller to actually observe _pool as None
        # before any of them finishes creating it — without this delay the
        # race window is too narrow to reliably exercise in a test.
        await asyncio.sleep(0.05)
        pool = object()
        created_pools.append(pool)
        return pool

    async def scenario():
        with patch("app.redis_pool.create_pool", side_effect=slow_create_pool):
            return await asyncio.gather(*(get_redis_pool() for _ in range(10)))

    results = asyncio.run(scenario())

    assert len(created_pools) == 1  # create_pool was only ever actually called once
    assert all(pool is results[0] for pool in results)  # every caller got the same instance
