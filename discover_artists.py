"""
Spotify artist discovery helpers.
"""
# Run `pip install -r requirements.txt` before executing this script.

import asyncio
import itertools
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

from playwright.async_api import Page, async_playwright

ALPHABET = tuple("abcdefghijklmnopqrstuvwxyz0123456789")
SEARCH_BASE_URL = "https://open.spotify.com/search/"
ARTIST_URL_PREFIX = "https://open.spotify.com"
ARTIST_HREF_RE = re.compile(r"^/artist/([A-Za-z0-9]+)$")


@dataclass
class DiscoverConfig:
    target_count: int
    per_query_target: int = 320
    max_scrolls: int = 75
    scroll_pause_ms: int = 600
    initial_wait_ms: int = 1500
    deterministic_length_limit: int = 3
    random_length_min: int = 4
    random_length_max: int = 7
    random_length_cap: int = 9
    random_batch_size: int = 900
    max_empty_queries: int = 1500
    empty_widen_interval: int = 120
    worker_count: int = 8
    headless: bool = True
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119 Safari/537.36"
    )


class QueryPlanner:
    """Generate varied search queries, broadening the search space over time."""

    def __init__(
        self,
        *,
        alphabet: Iterable[str],
        deterministic_limit: int,
        random_min: int,
        random_max: int,
        random_cap: int,
        batch_size: int,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._alphabet = tuple(alphabet)
        self._deterministic_limit = max(0, deterministic_limit)
        self._random_min = max(1, random_min)
        self._random_max = max(self._random_min, random_max)
        self._random_cap = max(self._random_max, random_cap)
        self._batch_size = max(1, batch_size)
        self._rng = rng or random.Random()
        self._queue: List[str] = []
        self._seen: Set[str] = set()
        self._length = 1
        self._deterministic_iter = (
            self._make_deterministic_iter(self._length)
            if self._length <= self._deterministic_limit
            else None
        )

    def __iter__(self) -> "QueryPlanner":
        return self

    def __next__(self) -> str:
        while True:
            if not self._queue:
                self._fill_queue()
            if not self._queue:
                continue
            candidate = self._queue.pop()
            if candidate in self._seen:
                continue
            self._seen.add(candidate)
            return candidate

    def widen_random_space(self) -> None:
        """Allow the planner to explore longer random strings."""
        if self._random_max < self._random_cap:
            self._random_max += 1
        if self._random_min < self._random_max - 1:
            self._random_min += 1

    def _fill_queue(self) -> None:
        if self._deterministic_iter is not None:
            chunk = list(itertools.islice(self._deterministic_iter, self._batch_size))
            if not chunk:
                self._advance_length()
                return
            self._rng.shuffle(chunk)
            self._queue.extend(chunk)
            return

        bucket: Set[str] = set()
        while len(bucket) < self._batch_size:
            length = self._rng.randint(self._random_min, self._random_max)
            bucket.add("".join(self._rng.choices(self._alphabet, k=length)))
        randomized = list(bucket)
        self._rng.shuffle(randomized)
        self._queue.extend(randomized)

    def _advance_length(self) -> None:
        self._length += 1
        if self._length <= self._deterministic_limit:
            self._deterministic_iter = self._make_deterministic_iter(self._length)
        else:
            self._deterministic_iter = None

    @staticmethod
    def _make_deterministic_iter(length: int):
        for combo in itertools.product(ALPHABET, repeat=length):
            yield "".join(combo)


class SharedScrapeState:
    """Shared mutable state accessed by concurrent scraping workers."""

    def __init__(self, config: DiscoverConfig, planner: QueryPlanner, seen: Set[str]) -> None:
        self.config = config
        self.planner = planner
        self.seen = seen
        self.planner_lock = asyncio.Lock()
        self.state_lock = asyncio.Lock()
        self.stop_event = asyncio.Event()
        self.empty_queries = 0

    async def next_query(self) -> str:
        async with self.planner_lock:
            return next(self.planner)

    async def record_failure(self) -> int:
        async with self.state_lock:
            self.empty_queries += 1
            return self.empty_queries

    async def record_batch(self, batch: Set[str]) -> Tuple[int, int, int]:
        async with self.state_lock:
            additions = [url for url in batch if url not in self.seen]
            if additions:
                self.seen.update(additions)
                self.empty_queries = 0
            else:
                self.empty_queries += 1
            total = len(self.seen)
            empty_now = self.empty_queries
        return total, len(additions), empty_now

    async def handle_stall(self, empty_now: int) -> None:
        if (
            empty_now
            and self.config.empty_widen_interval
            and empty_now % self.config.empty_widen_interval == 0
        ):
            async with self.planner_lock:
                self.planner.widen_random_space()
        if empty_now >= self.config.max_empty_queries:
            if not self.stop_event.is_set():
                print("[info] No new artists discovered for a while; stopping early.", flush=True)
            self.stop_event.set()


async def collect_from_query(
    page: Page,
    *,
    query: str,
    per_query_target: int,
    max_scrolls: int,
    scroll_pause_ms: int,
    initial_wait_ms: int,
) -> Set[str]:
    """Visit a Spotify search query and extract artist URLs."""
    await page.goto(SEARCH_BASE_URL + query, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(initial_wait_ms)

    collected: Set[str] = set()
    last_height = 0
    for _ in range(max_scrolls):
        anchors = await page.locator('a[href^="/artist/"]').evaluate_all(
            "els => els.map(e => e.getAttribute('href'))"
        )
        for href in anchors:
            if href and ARTIST_HREF_RE.match(href):
                collected.add(ARTIST_URL_PREFIX + href)
        if len(collected) >= per_query_target:
            break

        await page.evaluate("window.scrollBy(0, document.body.scrollHeight);")
        await page.wait_for_timeout(scroll_pause_ms)

        height = await page.evaluate("document.body.scrollHeight")
        if height == last_height:
            break
        last_height = height

    return collected


async def _worker_loop(worker_id: int, shared: SharedScrapeState, browser, config: DiscoverConfig) -> None:
    context = await browser.new_context(user_agent=config.user_agent)
    page = await context.new_page()
    try:
        while not shared.stop_event.is_set():
            query = await shared.next_query()
            try:
                batch = await collect_from_query(
                    page,
                    query=query,
                    per_query_target=config.per_query_target,
                    max_scrolls=config.max_scrolls,
                    scroll_pause_ms=config.scroll_pause_ms,
                    initial_wait_ms=config.initial_wait_ms,
                )
            except Exception as exc:
                print(f"[warn] worker={worker_id} query='{query}' failed: {exc}", flush=True)
                empty_now = await shared.record_failure()
                await shared.handle_stall(empty_now)
                continue

            total, added, empty_now = await shared.record_batch(batch)
            if added:
                print(
                    f"[{time.strftime('%H:%M:%S')}] worker={worker_id} +{added} "
                    f"(total {total}/{config.target_count}) via '{query}'",
                    flush=True,
                )
                if total >= config.target_count:
                    shared.stop_event.set()
                    break
                continue

            await shared.handle_stall(empty_now)
            if shared.stop_event.is_set():
                break
    finally:
        await context.close()


async def discover(config: DiscoverConfig, *, initial_seen: Optional[Set[str]] = None) -> List[str]:
    """Discover Spotify artist URLs until the target count is reached."""
    seen: Set[str] = set(initial_seen or ())
    planner = QueryPlanner(
        alphabet=ALPHABET,
        deterministic_limit=config.deterministic_length_limit,
        random_min=config.random_length_min,
        random_max=config.random_length_max,
        random_cap=config.random_length_cap,
        batch_size=config.random_batch_size,
    )
    shared = SharedScrapeState(config, planner, seen)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=config.headless)
        try:
            if shared.seen:
                remaining = max(0, config.target_count - len(shared.seen))
                print(
                    f"[info] Resuming with {len(shared.seen)} pre-existing artist URLs. Need {remaining} more.",
                    flush=True,
                )
                if remaining <= 0:
                    return list(shared.seen)[: config.target_count]

            worker_total = max(1, config.worker_count)
            tasks = [
                asyncio.create_task(_worker_loop(idx, shared, browser, config))
                for idx in range(worker_total)
            ]
            await asyncio.gather(*tasks)
        finally:
            await browser.close()

    return list(shared.seen)[: config.target_count]


def load_existing_urls(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(ARTIST_URL_PREFIX)
    }


def save_urls(path: Path, urls: Iterable[str]) -> None:
    path.write_text("\n".join(sorted(urls)), encoding="utf-8")


def build_config(args) -> DiscoverConfig:
    return DiscoverConfig(
        target_count=args.count,
        per_query_target=args.per_query_target,
        max_scrolls=args.max_scrolls,
        scroll_pause_ms=args.scroll_pause_ms,
        initial_wait_ms=args.initial_wait_ms,
        deterministic_length_limit=args.deterministic_length,
        random_length_min=args.random_min,
        random_length_max=args.random_max,
        random_length_cap=args.random_cap,
        random_batch_size=args.batch_size,
        max_empty_queries=args.max_empty_queries,
        empty_widen_interval=args.empty_widen_interval,
        worker_count=args.workers,
        headless=not args.headed,
    )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Discover Spotify artist links without the official API."
    )
    parser.add_argument("--count", type=int, default=1000, help="Total artist URLs to collect.")
    parser.add_argument("--out", type=str, default="urls.txt", help="File to store the collected URLs.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing URLs in the output file (default is to resume).",
    )
    parser.add_argument("--headed", action="store_true", help="Run with a visible browser window.")
    parser.add_argument(
        "--per-query-target",
        type=int,
        default=320,
        help="Stop scrolling a query once this many unique artist URLs are found.",
    )
    parser.add_argument("--max-scrolls", type=int, default=75, help="Maximum scroll iterations per query.")
    parser.add_argument("--scroll-pause-ms", type=int, default=600, help="Delay between scrolls in milliseconds.")
    parser.add_argument("--initial-wait-ms", type=int, default=1500, help="Wait after navigation before scraping.")
    parser.add_argument(
        "--deterministic-length",
        type=int,
        default=3,
        help="Generate every possible query up to this length before switching to random queries.",
    )
    parser.add_argument(
        "--random-min",
        type=int,
        default=4,
        help="Smallest length for random queries after deterministic ones are exhausted.",
    )
    parser.add_argument(
        "--random-max",
        type=int,
        default=7,
        help="Initial maximum length for random queries.",
    )
    parser.add_argument(
        "--random-cap",
        type=int,
        default=9,
        help="Upper bound for automatic query length widening when scraping stalls.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=900,
        help="Number of queries to enqueue at a time.",
    )
    parser.add_argument(
        "--max-empty-queries",
        type=int,
        default=1500,
        help="Abort after this many consecutive queries without discovering a new artist.",
    )
    parser.add_argument(
        "--empty-widen-interval",
        type=int,
        default=120,
        help="After this many empty queries, expand the random query length range.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel Playwright contexts to run.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.out)
    existing = set()
    if not args.fresh:
        existing = load_existing_urls(output_path)
        if existing:
            print(f"[info] Loaded {len(existing)} existing artist URLs from {output_path}", flush=True)

    config = build_config(args)
    urls = asyncio.run(discover(config, initial_seen=existing))
    save_urls(output_path, urls)
    print(f"[done] Stored {len(urls)} artist URLs in {output_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[info] Interrupted by user.", flush=True)
