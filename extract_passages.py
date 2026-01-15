import json
with open("train.jsonl", "r", encoding="utf-8") as infile, open("train_passages.jsonl", "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        passage = data["chosen"][1]["content"]
        outfile.write(json.dumps({"passage": passage}) + "\n")
