import json
import random

INPUT_FILE = "refined_passages.jsonl"
OUTPUT_FILE = "refined_passages_chatml.jsonl"


PROMPT_PREFIXES = [
    "Rewrite this AI passage to sound more humanlike:",
    "Rewrite this AI passage so it reads more human:",
    "Make this AI passage sound more human:",
    "Polish this AI passage to feel more human:",
    "Rewrite this AI-generated passage in a more human voice:",
    "Rewrite this passage to sound more natural and human:",
    "Improve this AI passage to sound more humanlike:",
    "Rephrase this AI passage to feel more human:",
    "Rewrite this passage so it reads like a human wrote it:",
    "Convert this AI passage into a more human-sounding version:",
    "Rewrite this AI passage to be more human and natural:",
    "Make this passage sound less AI and more human:",
    "Rewrite this passage in a more human style:",
    "Rewrite this AI passage to feel more natural:",
    "Rewrite this passage so it sounds more humanlike:",
]


def to_chatml(refined_passage: str, true_passage: str) -> dict:
    prefix = random.choice(PROMPT_PREFIXES)
    return {
        "messages": [
            {
                "role": "user",
                "content": f"{prefix}\n{refined_passage}",
            },
            {"role": "assistant", "content": true_passage},
        ]
    }


def main() -> None:
    with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(
        OUTPUT_FILE, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            refined = obj.get("refined_passage")
            true = obj.get("true_passage")
            if refined is None or true is None:
                continue
            outfile.write(json.dumps(to_chatml(refined, true)) + "\n")


if __name__ == "__main__":
    main()
