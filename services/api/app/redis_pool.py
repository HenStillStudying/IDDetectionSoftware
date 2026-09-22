"""Lazy singleton Redis connection pool used to enqueue/poll async jobs.

A plain module-level singleton rather than FastAPI lifespan state (and not
@lru_cache, since pool creation is async) — kept behind a dependency
function so tests can override it with a fake pool (see services/ocr's
get_ocr_service for the same pattern) instead of needing a real Redis
server.
"""

from __future__ import annotations

import asyncio

from arq.connections import ArqRedis, RedisSettings, create_pool

from .config import settings

_pool: ArqRedis | None = None
_pool_lock = asyncio.Lock()


async def get_redis_pool() -> ArqRedis:
    global _pool
    # Fast path: once created, every caller just reads the existing pool,
    # no lock needed. The lock only guards the narrow window before that —
    # without it, two requests arriving before the pool exists could both
    # pass this None check, both call create_pool(), and whichever finished
    # last would silently overwrite the other's pool, leaking its
    # connections. The re-check inside the lock is what actually closes
    # that window (the first waiter to acquire it creates the pool; every
    # other waiter then sees it's no longer None and skips creating another).
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is None:
            _pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    return _pool
