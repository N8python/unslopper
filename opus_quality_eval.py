import argparse
import asyncio
import json
import os
import re
from typing import Optional

from openai import AsyncOpenAI

DEFAULT_INPUT_FILE = "unslopped_stories.jsonl"
DEFAULT_OUTPUT_FILE = "unslopped_stories_quality_te.jsonl"
MODEL = "anthropic/claude-opus-4.5"
CONCURRENCY = 32
MAX_TOKENS = 900

SYSTEM_PROMPT = (
    "You are a rigorous literary critic. Analyze the story in depth and then score it. "
    "Return only XML tags with no extra text."
)


def load_stories(path: str) -> list[dict]:
    stories = []
    with open(path, "r", encoding="utf-8") as infile:
        for idx, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            original_story = obj.get("original_story")
            unslopped_story = obj.get("unslopped_story")
            if original_story is not None and unslopped_story is not None:
                stories.append(
                    {
                        "story_id": idx,
                        "kind": "pair",
                        "original_story": original_story,
                        "unslopped_story": unslopped_story,
                    }
                )
                continue
            story = obj.get("story")
            if story is not None:
                stories.append(
                    {
                        "story_id": idx,
                        "kind": "single",
                        "prompt_id": obj.get("prompt_id"),
                        "prompt": obj.get("prompt"),
                        "story": story,
                    }
                )
    return stories


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


def is_complete(record: Optional[dict], story: dict) -> bool:
    if not record:
        return False
    if story["kind"] == "pair":
        if record.get("original_story") != story["original_story"]:
            return False
        if record.get("unslopped_story") != story["unslopped_story"]:
            return False
        original_scores = record.get("original_eval", {}).get("scores", {})
        unslopped_scores = record.get("unslopped_eval", {}).get("scores", {})
        for scores in (original_scores, unslopped_scores):
            if any(scores.get(k) is None for k in ("coherence", "style", "general")):
                return False
        return True
    if story["kind"] == "single":
        if record.get("story") != story["story"]:
            return False
        scores = record.get("story_eval", {}).get("scores", {})
        if any(scores.get(k) is None for k in ("coherence", "style", "general")):
            return False
        return True
    return False


def write_results(path: str, stories: list[dict], results: dict[int, dict]) -> None:
    with open(path, "w", encoding="utf-8") as outfile:
        for story in stories:
            story_id = story["story_id"]
            if story["kind"] == "pair":
                record = results.get(story_id) or {
                    "story_id": story_id,
                    "original_story": story["original_story"],
                    "unslopped_story": story["unslopped_story"],
                }
            else:
                record = results.get(story_id) or {
                    "story_id": story_id,
                    "prompt_id": story.get("prompt_id"),
                    "prompt": story.get("prompt"),
                    "story": story["story"],
                }
            outfile.write(json.dumps(record) + "\n")


async def process_story(
    client: AsyncOpenAI,
    story: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    story_id = story["story_id"]
    try:
        if story["kind"] == "pair":
            original_story = story["original_story"]
            unslopped_story = story["unslopped_story"]
            original_eval, unslopped_eval = await asyncio.gather(
                evaluate_story(client, original_story, semaphore),
                evaluate_story(client, unslopped_story, semaphore),
            )
            record = {
                "story_id": story_id,
                "original_story": original_story,
                "unslopped_story": unslopped_story,
                "original_eval": original_eval,
                "unslopped_eval": unslopped_eval,
            }
        else:
            story_text = story["story"]
            story_eval = await evaluate_story(client, story_text, semaphore)
            record = {
                "story_id": story_id,
                "prompt_id": story.get("prompt_id"),
                "prompt": story.get("prompt"),
                "story": story_text,
                "story_eval": story_eval,
            }
    except Exception as exc:
        record = {
            "story_id": story_id,
            "prompt_id": story.get("prompt_id"),
            "prompt": story.get("prompt"),
            "original_story": story.get("original_story"),
            "unslopped_story": story.get("unslopped_story"),
            "story": story.get("story"),
            "error": str(exc),
        }
    print(f"Completed {story_id}")
    return record


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate story quality with Claude Opus via OpenRouter."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()

    api_key = os.environ.get(
        "OPENROUTER_API_KEY",
        "API_KEY_HERE",  
    )

    stories = load_stories(args.input)
    if not stories:
        raise RuntimeError(f"No stories found in {args.input}")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    results = load_existing_results(args.output)
    missing = [story for story in stories if not is_complete(results.get(story["story_id"]), story)]

    if not missing:
        print("All stories already evaluated.")
        return

    tasks = [
        asyncio.create_task(process_story(client, story, semaphore))
        for story in missing
    ]
    for record in await asyncio.gather(*tasks):
        results[record["story_id"]] = record

    write_results(args.output, stories, results)


if __name__ == "__main__":
    asyncio.run(main())
