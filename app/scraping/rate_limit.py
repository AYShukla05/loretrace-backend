import asyncio
import time
from urllib.parse import urlparse


class DomainRateLimiter:
    def __init__(self, min_interval_seconds: float = 2.0) -> None:
        self._min_interval = min_interval_seconds
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_request: dict[str, float] = {}

    def _lock_for(self, domain: str) -> asyncio.Lock:
        if domain not in self._locks:
            self._locks[domain] = asyncio.Lock()
        return self._locks[domain]

    async def wait(self, url: str) -> None:
        domain = urlparse(url).netloc
        async with self._lock_for(domain):
            last = self._last_request.get(domain)
            if last is not None:
                remaining = self._min_interval - (time.monotonic() - last)
                if remaining > 0:
                    await asyncio.sleep(remaining)
            self._last_request[domain] = time.monotonic()
