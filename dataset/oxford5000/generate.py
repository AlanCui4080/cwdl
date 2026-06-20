import json

with open("dataset/full-word.json", "r", encoding="utf-8") as f:
    data = json.load(f)

words = [item["value"]["word"] for item in data if len(item["value"]["word"]) >= 3]

with open("words.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(words) + "\n")

print(f"Extracted {len(words)} words (>=3 letters) to words.txt")
