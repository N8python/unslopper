from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

model, tokenizer = load("Unslopper-30B-A3B-6bit")

def build_input(passage: str) -> str:
    return f"Rewrite this AI passage to sound more humanlike:\n{passage}"

def unslop(passage: str) -> str:
    input_text = build_input(passage)
    messages = [
        {"role": "user", "content": input_text}
    ]
    output = generate(
        model,
        tokenizer,
        tokenizer.apply_chat_template(messages, add_generation_prompt=True),
        max_tokens=4096,
        sampler=make_sampler(temp=0.8),
        logits_processors=make_logits_processors(repetition_penalty=1.1),
        verbose=True
    )
    return output.strip()

import json
INPUT_FILE = "short_stories.jsonl"
OUTPUT_FILE = "unslopped_stories.jsonl"

def main() -> None:
    with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(
        OUTPUT_FILE, "w", encoding="utf-8"
    ) as outfile:
        done = 0
        for line in infile:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            story = obj.get("story")
            if story is None:
                continue
            unslopped_story = unslop(story)
            done += 1
            print(f"Processed {done} stories", end="\r")
            outfile.write(json.dumps({"original_story": story, "unslopped_story": unslopped_story}) + "\n")
if __name__ == "__main__":
    main()