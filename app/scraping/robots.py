from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

_robots_cache: dict[str, RobotFileParser] = {}


async def _get_robots_parser(client: httpx.AsyncClient, url: str) -> RobotFileParser:
    origin = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    if origin in _robots_cache:
        return _robots_cache[origin]

    parser = RobotFileParser()
    try:
        response = await client.get(f"{origin}/robots.txt", timeout=10)
        response.raise_for_status()
        parser.parse(response.text.splitlines())
    except httpx.HTTPError:
        # robots.txt unreachable: fail open, treat as unrestricted
        parser.parse([])

    _robots_cache[origin] = parser
    return parser


async def can_fetch(client: httpx.AsyncClient, url: str, user_agent: str) -> bool:
    parser = await _get_robots_parser(client, url)
    return parser.can_fetch(user_agent, url)
