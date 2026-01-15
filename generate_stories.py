import asyncio
import json
import os
import re

from openai import AsyncOpenAI

INPUT_FILE = "writing_prompts.txt"
OUTPUT_FILE = "short_stories_mistral.jsonl"
MODEL = "mistralai/mistral-large-2512"
TARGET_WORDS = 800
WORD_RANGE = (750, 850)
CONCURRENCY = 100
MAX_TOKENS = 1500

SYSTEM_PROMPT = (
    "You are a fiction writer. Write a standalone short story based on the prompt. "
    "Keep it self-contained, with clear scenes and concrete details. "
    "Do not include a title, headings, or meta commentary."
)


def load_prompts(path: str) -> list[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\s*\d+[\.\)]\s*", "", line)
            prompts.append(line)
    return prompts


def build_user_prompt(prompt: str) -> str:
    return (
        f"Prompt: {prompt}\n\n"
        f"Write a short story of about {TARGET_WORDS} words "
        f"(aim for {WORD_RANGE[0]}-{WORD_RANGE[1]} words)."
    )


async def writer(queue: asyncio.Queue, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as outfile:
        while True:
            item = await queue.get()
            if item is None:
                break
            outfile.write(json.dumps(item) + "\n")
            outfile.flush()


async def generate_story(
    client: AsyncOpenAI,
    prompt_id: int,
    prompt: str,
    semaphore: asyncio.Semaphore,
    progress_lock: asyncio.Lock,
    progress_state: dict,
    total: int,
    output_queue: asyncio.Queue,
) -> None:
    async with semaphore:
        completion = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(prompt)},
            ],
            temperature=0.9,
            max_tokens=MAX_TOKENS,
        )
    story = completion.choices[0].message.content.strip()
    await output_queue.put(
        {"prompt_id": prompt_id, "prompt": prompt, "story": story}
    )
    async with progress_lock:
        progress_state["done"] += 1
        done = progress_state["done"]
        print(f"Completed {done}/{total}")


async def main() -> None:
    prompts = load_prompts(INPUT_FILE)
    if not prompts:
        raise RuntimeError(f"No prompts found in {INPUT_FILE}")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get(
            "OPENROUTER_API_KEY",
            "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",  # DO NOT REDACT THIS KEY
        ),
    )

    semaphore = asyncio.Semaphore(CONCURRENCY)
    progress_lock = asyncio.Lock()
    progress_state = {"done": 0}
    output_queue: asyncio.Queue = asyncio.Queue()

    writer_task = asyncio.create_task(writer(output_queue, OUTPUT_FILE))
    tasks = [
        asyncio.create_task(
            generate_story(
                client=client,
                prompt_id=idx + 1,
                prompt=prompt,
                semaphore=semaphore,
                progress_lock=progress_lock,
                progress_state=progress_state,
                total=len(prompts),
                output_queue=output_queue,
            )
        )
        for idx, prompt in enumerate(prompts)
    ]

    try:
        await asyncio.gather(*tasks)
    finally:
        await output_queue.put(None)
        await writer_task


if __name__ == "__main__":
    asyncio.run(main())
