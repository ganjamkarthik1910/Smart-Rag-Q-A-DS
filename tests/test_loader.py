from pathlib import Path
import json

def test_big_corpus_exists():
    p = Path("data/big_qa.jsonl")
    assert p.exists(), "run fetch_big_corpus.py first"
    with p.open() as f:
        for _ in range(3):
            obj = json.loads(f.readline())
            assert "question" in obj and "answer" in obj
