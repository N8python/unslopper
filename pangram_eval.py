import asyncio
import json
import os
from typing import Optional
import urllib.request

API_URL = "https://text.api.pangram.com/v3"
INPUT_FILE = "unslopped_stories.jsonl"
OUTPUT_FILE = "unslopped_stories_pangram.jsonl"
CONCURRENCY = 8
REQUEST_TIMEOUT = 60
MAX_PASSES = int(os.environ.get("PANGRAM_MAX_PASSES", "0"))
RETRY_DELAY_SECONDS = 2


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
            if original_story is None or unslopped_story is None:
                continue
            stories.append(
                {
                    "story_id": idx,
                    "original_story": original_story,
                    "unslopped_story": unslopped_story,
                }
            )
    return stories


def pangram_request(text: str, api_key: str) -> dict:
    payload = {"text": text, "public_dashboard_link": False}
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(API_URL, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("x-api-key", api_key)

    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return json.load(response)


async def analyze_text(
    text: str, api_key: str, semaphore: asyncio.Semaphore
) -> dict:
    async with semaphore:
        return await asyncio.to_thread(pangram_request, text, api_key)


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
    if record.get("original_story") != story["original_story"]:
        return False
    if record.get("unslopped_story") != story["unslopped_story"]:
        return False
    original = record.get("original_pangram")
    unslopped = record.get("unslopped_pangram")
    if not original or not unslopped:
        return False
    if "fraction_ai" not in original or "fraction_ai" not in unslopped:
        return False
    return True


async def process_story(
    story: dict,
    api_key: str,
    semaphore: asyncio.Semaphore,
) -> None:
    story_id = story["story_id"]
    original_story = story["original_story"]
    unslopped_story = story["unslopped_story"]
    try:
        original_resp, unslopped_resp = await asyncio.gather(
            analyze_text(original_story, api_key, semaphore),
            analyze_text(unslopped_story, api_key, semaphore),
        )
        record = {
            "story_id": story_id,
            "original_story": original_story,
            "unslopped_story": unslopped_story,
            "original_pangram": original_resp,
            "unslopped_pangram": unslopped_resp,
        }
    except Exception as exc:
        record = {
            "story_id": story_id,
            "original_story": original_story,
            "unslopped_story": unslopped_story,
            "error": str(exc),
        }
    print(f"Completed {story_id}")
    return record


def write_results(path: str, stories: list[dict], results: dict[int, dict]) -> None:
    with open(path, "w", encoding="utf-8") as outfile:
        for story in stories:
            story_id = story["story_id"]
            record = results.get(story_id) or {
                "story_id": story_id,
                "original_story": story["original_story"],
                "unslopped_story": story["unslopped_story"],
            }
            outfile.write(json.dumps(record) + "\n")


async def main() -> None:
    api_key = os.environ.get("PANGRAM_API_KEY")
    if not api_key:
        raise SystemExit("Set PANGRAM_API_KEY to your Pangram API key.")

    stories = load_stories(INPUT_FILE)
    if not stories:
        raise RuntimeError(f"No stories found in {INPUT_FILE}")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = load_existing_results(OUTPUT_FILE)

    pass_count = 0
    while True:
        missing = [
            story
            for story in stories
            if not is_complete(results.get(story["story_id"]), story)
        ]
        if not missing:
            break

        pass_count += 1
        print(f"Missing {len(missing)} stories; pass {pass_count}")
        tasks = [
            asyncio.create_task(process_story(story, api_key, semaphore))
            for story in missing
        ]
        for record in await asyncio.gather(*tasks):
            results[record["story_id"]] = record

        write_results(OUTPUT_FILE, stories, results)

        if MAX_PASSES and pass_count >= MAX_PASSES:
            print("Reached PANGRAM_MAX_PASSES limit.")
            break
        if RETRY_DELAY_SECONDS > 0:
            await asyncio.sleep(RETRY_DELAY_SECONDS)

    write_results(OUTPUT_FILE, stories, results)


if __name__ == "__main__":
    asyncio.run(main())
