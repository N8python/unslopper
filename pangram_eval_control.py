import asyncio
import json
import os
from typing import Optional
import urllib.request

API_URL = "https://text.api.pangram.com/v3"
CONTROL_FILE = "unslopped_stories_control.jsonl"
PANGRAM_FILE = "unslopped_stories_pangram.jsonl"
OUTPUT_FILE = os.environ.get("PANGRAM_CONTROL_OUTPUT", PANGRAM_FILE)
CONCURRENCY = 8
REQUEST_TIMEOUT = 60
MAX_PASSES = int(os.environ.get("PANGRAM_MAX_PASSES", "0"))
RETRY_DELAY_SECONDS = 2


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


def load_pangram_records(path: str) -> dict[int, dict]:
    if not os.path.exists(path):
        return {}
    records = {}
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            story_id = obj.get("story_id")
            if isinstance(story_id, int):
                records[story_id] = obj
    return records


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


def is_complete(record: Optional[dict], control_story: str) -> bool:
    if not record:
        return False
    if record.get("control_story") != control_story:
        return False
    control = record.get("control_pangram")
    if not control or "fraction_ai" not in control:
        return False
    return True


def sync_control_story(record: dict, control_story: str) -> None:
    if record.get("control_story") != control_story:
        record.pop("control_pangram", None)
        record.pop("control_error", None)
    record["control_story"] = control_story


async def process_story(
    story_id: int,
    control_story: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
) -> tuple[int, dict]:
    try:
        control_resp = await analyze_text(control_story, api_key, semaphore)
        update = {"control_story": control_story, "control_pangram": control_resp}
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
    api_key = os.environ.get("PANGRAM_API_KEY")
    if not api_key:
        raise SystemExit("Set PANGRAM_API_KEY to your Pangram API key.")

    control_stories = load_control_stories(CONTROL_FILE)
    if not control_stories:
        raise RuntimeError(f"No control stories found in {CONTROL_FILE}")

    records = load_pangram_records(PANGRAM_FILE)
    if not records:
        raise RuntimeError(f"No pangram records found in {PANGRAM_FILE}")

    story_ids = sorted(control_stories.keys())
    missing = []
    for story_id in story_ids:
        record = records.get(story_id) or {"story_id": story_id}
        sync_control_story(record, control_stories[story_id])
        records[story_id] = record
        if not is_complete(record, control_stories[story_id]):
            missing.append(story_id)

    semaphore = asyncio.Semaphore(CONCURRENCY)
    pass_count = 0
    while missing:
        pass_count += 1
        print(f"Missing {len(missing)} control stories; pass {pass_count}")
        tasks = [
            asyncio.create_task(
                process_story(story_id, control_stories[story_id], api_key, semaphore)
            )
            for story_id in missing
        ]
        for story_id, update in await asyncio.gather(*tasks):
            record = records.get(story_id) or {"story_id": story_id}
            record.update(update)
            records[story_id] = record

        write_results(OUTPUT_FILE, story_ids, records)

        if MAX_PASSES and pass_count >= MAX_PASSES:
            print("Reached PANGRAM_MAX_PASSES limit.")
            break
        if RETRY_DELAY_SECONDS > 0:
            await asyncio.sleep(RETRY_DELAY_SECONDS)

        missing = [
            story_id
            for story_id in story_ids
            if not is_complete(records.get(story_id), control_stories[story_id])
        ]

    write_results(OUTPUT_FILE, story_ids, records)


if __name__ == "__main__":
    asyncio.run(main())
