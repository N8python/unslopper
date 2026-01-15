import argparse
import asyncio
import json
import os
import re
from typing import Optional

from openai import AsyncOpenAI

DEFAULT_CONTROL_FILE = "unslopped_stories_control.jsonl"
DEFAULT_QUALITY_FILE = "unslopped_stories_quality.jsonl"
MODEL = "anthropic/claude-opus-4.5"
CONCURRENCY = 32
MAX_TOKENS = 900

SYSTEM_PROMPT = (
    "You are a rigorous literary critic. Analyze the story in depth and then score it. "
    "Return only XML tags with no extra text."
)


def load_control_stories(path: str) -> dict[int, str]:
    stories = {}
    with open(path, "r", encoding="utf-8") as infile:
        for idx, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            control_story = obj.get("unslopped_story")
            if control_story is None:
                continue
            stories[idx] = control_story
    return stories


def load_existing_results(path: str) -> dict[int, dict]:
    if not os.path.exists(path):
        return {}
    results = {}
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            story_id = obj.get("story_id")
            if isinstance(story_id, int):
                results[story_id] = obj
    return results


def build_user_prompt(story: str) -> str:
    return (
        "Analyze the story deeply and place your analysis inside <analysis> tags. "
        "Then provide three numeric scores from 1 to 10 (10 is best) using XML tags: "
        "<coherence>, <style>, and <general>. Return only these XML tags in order.\n\n"
        "<story>\n"
        f"{story}\n"
        "</story>"
    )


def extract_tag(text: str, tag: str) -> Optional[str]:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def extract_score(text: str, tag: str) -> Optional[float]:
    content = extract_tag(text, tag)
    if content is None:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", content)
    if not match:
        return None
    return float(match.group(1))


def parse_response(text: str) -> dict:
    analysis = extract_tag(text, "analysis")
    coherence = extract_score(text, "coherence")
    style = extract_score(text, "style")
    general = extract_score(text, "general")
    missing = []
    if coherence is None:
        missing.append("coherence")
    if style is None:
        missing.append("style")
    if general is None:
        missing.append("general")
    return {
        "analysis": analysis,
        "scores": {
            "coherence": coherence,
            "style": style,
            "general": general,
        },
        "missing_tags": missing,
    }


async def evaluate_story(
    client: AsyncOpenAI,
    story: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        completion = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(story)},
            ],
            temperature=0.2,
            max_tokens=MAX_TOKENS,
        )
    content = completion.choices[0].message.content.strip()
    parsed = parse_response(content)
    return {
        "raw_response": content,
        "analysis": parsed["analysis"],
        "scores": parsed["scores"],
        "missing_tags": parsed["missing_tags"],
    }


def is_complete(record: Optional[dict], control_story: str) -> bool:
    if not record:
        return False
    if record.get("control_story") != control_story:
        return False
    scores = record.get("control_eval", {}).get("scores", {})
    if any(scores.get(k) is None for k in ("coherence", "style", "general")):
        return False
    return True


def sync_control_story(record: dict, control_story: str) -> None:
    if record.get("control_story") != control_story:
        record.pop("control_eval", None)
        record.pop("control_error", None)
    record["control_story"] = control_story


async def process_story(
    client: AsyncOpenAI,
    story_id: int,
    control_story: str,
    semaphore: asyncio.Semaphore,
) -> tuple[int, dict]:
    try:
        control_eval = await evaluate_story(client, control_story, semaphore)
        update = {"control_story": control_story, "control_eval": control_eval}
    except Exception as exc:
        update = {"control_story": control_story, "control_error": str(exc)}
    print(f"Completed control {story_id}")
    return story_id, update


def write_results(path: str, story_ids: list[int], records: dict[int, dict]) -> None:
    with open(path, "w", encoding="utf-8") as outfile:
        for story_id in story_ids:
            record = records.get(story_id) or {"story_id": story_id}
            outfile.write(json.dumps(record) + "\n")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate control story quality with Claude Opus via OpenRouter."
    )
    parser.add_argument("--control", default=DEFAULT_CONTROL_FILE)
    parser.add_argument("--quality", default=DEFAULT_QUALITY_FILE)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output_path = args.output or args.quality

    api_key = os.environ.get(
        "OPENROUTER_API_KEY",
        "API_KEY_HERE",  
    )

    control_stories = load_control_stories(args.control)
    if not control_stories:
        raise RuntimeError(f"No control stories found in {args.control}")

    records = load_existing_results(args.quality)
    if not records:
        raise RuntimeError(f"No quality records found in {args.quality}")

    story_ids = sorted(set(records.keys()) | set(control_stories.keys()))
    missing = []
    for story_id in story_ids:
        record = records.get(story_id) or {"story_id": story_id}
        control_story = control_stories.get(story_id)
        if control_story is None:
            records[story_id] = record
            continue
        sync_control_story(record, control_story)
        records[story_id] = record
        if not is_complete(record, control_story):
            missing.append(story_id)

    if not missing:
        print("All control stories already evaluated.")
        write_results(output_path, story_ids, records)
        return

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        asyncio.create_task(
            process_story(client, story_id, control_stories[story_id], semaphore)
        )
        for story_id in missing
    ]
    for story_id, update in await asyncio.gather(*tasks):
        record = records.get(story_id) or {"story_id": story_id}
        record.update(update)
        records[story_id] = record

    write_results(output_path, story_ids, records)


if __name__ == "__main__":
    asyncio.run(main())
