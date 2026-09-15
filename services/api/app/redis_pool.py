"""Lazy singleton Redis connection pool used to enqueue/poll async jobs.

A plain module-level singleton rather than FastAPI lifespan state (and not
@lru_cache, since pool creation is async) — kept behind a dependency
function so tests can override it with a fake pool (see services/ocr's
get_ocr_service for the same pattern) instead of needing a real Redis
server.
"""

from __future__ import annotations

from arq.connections import ArqRedis, RedisSettings, create_pool

from .config import settings

_pool: ArqRedis | None = None


async def get_redis_pool() -> ArqRedis:
    global _pool
    if _pool is None:
        _pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    return _pool
