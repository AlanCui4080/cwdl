import random
import string

chars = string.ascii_uppercase + string.digits + " ?=/"
lines = set()
while len(lines) < 10000:
    n = random.randint(30, 50)
    s = ''.join(random.choices(chars, k=n))
    lines.add(s)

with open("random50.txt", "w") as f:
    for s in sorted(lines, key=lambda x: (len(x), x)):
        f.write(s + "\n")
print(f"Generated {len(lines)} unique alphanumeric strings (30-50 chars)")
