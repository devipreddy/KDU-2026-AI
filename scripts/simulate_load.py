from __future__ import annotations

import argparse
import asyncio
from statistics import mean
from time import perf_counter

import httpx

PROMPTS = [
    "Why was I charged again this month?",
    "I think I have a duplicate charge on my card ending 4242.",
    "Can you explain invoice INV-2026-0401?",
    "My payment failed. What should I do next?",
]


async def run_one(client: httpx.AsyncClient, base_url: str, user_index: int) -> float:
    started = perf_counter()
    prompt = PROMPTS[user_index % len(PROMPTS)]
    response = await client.post(
        f"{base_url}/api/demo/text-turn",
        json={"user_id": f"load-user-{user_index}", "transcript": prompt},
    )
    response.raise_for_status()
    _ = response.json()
    return perf_counter() - started


async def main(base_url: str, concurrency: int, total: int) -> None:
    async with httpx.AsyncClient(timeout=60.0) as client:
        latencies: list[float] = []
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_run(user_index: int) -> None:
            async with semaphore:
                latencies.append(await run_one(client, base_url, user_index))

        await asyncio.gather(*(bounded_run(i) for i in range(total)))
        print(f"completed={len(latencies)}")
        print(f"avg_ms={mean(latencies) * 1000:.1f}")
        print(f"max_ms={max(latencies) * 1000:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--total", type=int, default=100)
    args = parser.parse_args()
    asyncio.run(main(args.base_url, args.concurrency, args.total))
