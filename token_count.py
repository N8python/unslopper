from mlx_lm import load
import json
_, tokenizer = load("Qwen3-VL-Text-30B-A3B-Instruct-8bit")

def count_tokens(messages):
    return len(tokenizer.apply_chat_template(messages))

if __name__ == "__main__":
    with open("data/train.jsonl", "r", encoding="utf-8") as infile:
        percent_smaller_4096 = 0
        total = 0
        for line in infile:
            obj = json.loads(line)
            messages = obj["messages"]
            token_count = count_tokens(messages)
            #print(f"Token count: {token_count}")
            if token_count < 6144:
                percent_smaller_4096 += 1
            total += 1
        print(f"Percent smaller than 6144 tokens: {percent_smaller_4096 / total:.2%}")