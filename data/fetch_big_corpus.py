"""
Download & merge three openly licensed QA datasets into JSONL.
Creates data/big_qa.jsonl (~500 K rows after simple filters).
"""
import json, pathlib, datasets

OUT = pathlib.Path("data"); OUT.mkdir(parents=True, exist_ok=True)
file = (OUT / "big_qa.jsonl").open("w")

SOURCES = [
    "statsmath/data-stackexchange-2024-07",    # Data Stack Exchange
    "bigcode/stackoverflow-ml-2024-06",        # StackOverflow ML subset
    "bigcode/the-stack-ds-v1.2"                # The Stack DS slice
]

for src in SOURCES:
    print("→ loading", src)
    ds = datasets.load_dataset(src, split="train", streaming=True)
    for row in ds:
        q, a = row.get("question"), row.get("answer")
        if not q or not a or len(a) < 30:
            continue
        file.write(json.dumps({"question": q.strip(), "answer": a.strip()}) + "\n")

print("Done. Saved", file.name)
