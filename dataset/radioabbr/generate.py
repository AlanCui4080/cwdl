with open("radioabbr.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

words = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    words.append(line.split()[0])

with open("abbr.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(words) + "\n")

print(f"Extracted {len(words)} first-words to abbr.txt")
