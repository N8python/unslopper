import json
import os
import urllib.request

API_URL = "https://text.api.pangram.com/v3"


def analyze_text(text: str) -> dict:
    api_key = os.environ.get("PANGRAM_API_KEY")
    if not api_key:
        raise SystemExit("Set PANGRAM_API_KEY to your Pangram API key.")

    payload = {"text": text, "public_dashboard_link": False}
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(API_URL, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("x-api-key", api_key)

    with urllib.request.urlopen(request) as response:
        return json.load(response)


def main() -> None:
    sample_text = "The quick brown fox jumps over the lazy dog."
    result = analyze_text(sample_text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
