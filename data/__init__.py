"""
data package
~~~~~~~~~~~~
Utility helpers for loading, streaming and basic inspection of the
JSONL corpora used in Smart‑RAG‑Llama 3.
"""

from pathlib import Path
import json
from typing import Iterator, Dict, Any

__all__ = ["stream_jsonl", "count_lines"]

_DATA_ROOT = Path(__file__).parent

def stream_jsonl(filename: str) -> Iterator[Dict[str, Any]]:
    """
    Yields dictionaries from a JSONL file **relative to the data package**.

    Example
    -------
    >>> from data import stream_jsonl
    >>> for row in stream_jsonl("big_qa.jsonl"):
    ...     print(row["question"])
    ...     break
    """
    path = _DATA_ROOT / filename
    with path.open("r", encoding="utf‑8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def count_lines(filename: str) -> int:
    """Return number of non‑empty lines (i.e. records) in a JSONL file."""
    path = _DATA_ROOT / filename
    with path.open("r", encoding="utf‑8") as f:
        return sum(1 for line in f if line.strip())
