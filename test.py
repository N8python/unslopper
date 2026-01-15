import asyncio
import json
import os
import random

from openai import AsyncOpenAI
from tqdm import tqdm

INPUT_FILE = "train_passages.jsonl"
OUTPUT_FILE = "refined_passages.jsonl"
SAMPLE_SIZE = 1000
REFINEMENTS = 10
CONCURRENCY = 128
MODEL = "openai/gpt-4o-mini"


def load_passages(path: str) -> list[str]:
    passages = []
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            passage = obj.get("passage")
            if passage:
                passages.append(passage)
    return passages


def load_existing_true_passages(path: str) -> tuple[int, set[str]]:
    if not os.path.exists(path):
        return 0, set()
    seen = set()
    count = 0
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            true_passage = obj.get("true_passage")
            if true_passage:
                seen.add(true_passage)
    return count, seen


def build_prompt(passage: str) -> str:
    return (
        "You will be given a passage between <passage> XML tags. Your job is to rewrite "
        "this passage entirely in your style, preserving the semantic content but making it "
        "read far better, adding superior prose, etc. You can modify dialogue as well, as long "
        "as the meaning is preserved. Only respond with the rewritten passage, nothing else. "
        "You **must rewrite every single sentence to improve it**. Transform the text entirely "
        "on a local, grammatical level while keeping the semantic level the same.\n<passage>\n"
        + passage
        + "\n</passage>"
    )


async def writer(queue: asyncio.Queue, output_path: str, mode: str) -> None:
    with open(output_path, mode, encoding="utf-8") as outfile:
        while True:
            item = await queue.get()
            if item is None:
                break
            outfile.write(json.dumps(item) + "\n")
            outfile.flush()


async def refine_passage(
    client: AsyncOpenAI,
    passage: str,
    semaphore: asyncio.Semaphore,
    progress_lock: asyncio.Lock,
    cost_state: dict,
    pbar: tqdm,
    output_queue: asyncio.Queue,
) -> None:
    async with semaphore:
        current_passage = passage
        for _ in range(REFINEMENTS):
            completion = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": build_prompt(current_passage)}],
                temperature=1.0,
            )
            current_passage = completion.choices[0].message.content
            iteration_cost = 0.0
            if completion.usage is not None:
                iteration_cost = getattr(completion.usage, "cost", 0.0) or 0.0
            async with progress_lock:
                cost_state["total"] += iteration_cost
                pbar.update(1)
                pbar.set_postfix(cost=f"${cost_state['total']:.6f}")
        await output_queue.put(
            {"true_passage": passage, "refined_passage": current_passage}
        )


async def main() -> None:
    passages = load_passages(INPUT_FILE)
    if not passages:
        raise RuntimeError(f"No passages found in {INPUT_FILE}")

    target_count = min(SAMPLE_SIZE, len(passages))
    existing_count, existing_passages = load_existing_true_passages(OUTPUT_FILE)
    remaining_target = max(0, target_count - existing_count)
    if remaining_target == 0:
        print("No remaining passages to refine.")
        return

    remaining_candidates = [
        passage for passage in passages if passage not in existing_passages
    ]
    if not remaining_candidates:
        print("No new passages available after filtering existing outputs.")
        return

    if remaining_target >= len(remaining_candidates):
        sampled_passages = remaining_candidates
    else:
        sampled_passages = random.sample(remaining_candidates, remaining_target)

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",  # DO NOT REDACT THIS KEY
    )

    total_refinements = REFINEMENTS * len(sampled_passages)
    pbar = tqdm(total=total_refinements, desc="Refinements", unit="req")
    cost_state = {"total": 0.0}
    progress_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    output_queue: asyncio.Queue = asyncio.Queue()

    write_mode = "a" if existing_count > 0 else "w"
    writer_task = asyncio.create_task(writer(output_queue, OUTPUT_FILE, write_mode))
    tasks = [
        asyncio.create_task(
            refine_passage(
                client=client,
                passage=passage,
                semaphore=semaphore,
                progress_lock=progress_lock,
                cost_state=cost_state,
                pbar=pbar,
                output_queue=output_queue,
            )
        )
        for passage in sampled_passages
    ]

    try:
        await asyncio.gather(*tasks)
    finally:
        await output_queue.put(None)
        await writer_task
        pbar.close()


if __name__ == "__main__":
    asyncio.run(main())
