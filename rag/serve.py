"""
FastAPI + vLLM micro‑service with RAG.
Start with:  python rag/serve.py --ckpt outputs/llama3-ds-qlora
"""
import argparse, asyncio
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

app = FastAPI()

class Query(BaseModel):
    question: str
    top_k: int = 4

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to QLoRA model dir")
    p.add_argument("--collection", default="ds_qa_chunks")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    return p.parse_args()

# will be populated in main()
encoder: SentenceTransformer = None
client: QdrantClient = None
llm: LLM = None
COLLECTION: str = None

@app.post("/query")
async def query(q: Query):
    emb = encoder.encode(q.question)
    hits = client.search(COLLECTION, emb, limit=q.top_k)
    context = "\n\n".join([h.payload["answer"] for h in hits])
    prompt = (
        "You are an expert data‑science tutor. "
        "Use the context to answer factually and cite sources.\n\n"
        f"Context:\n{context}\n\nQuestion: {q.question}\nAnswer:"
    )
    out = await asyncio.to_thread(
        llm.generate, prompt, SamplingParams(temperature=0.2, top_p=0.95, max_tokens=256)
    )
    answer = out[0].outputs[0].text.strip()
    sources = [h.payload for h in hits]
    return {"answer": answer, "sources": sources}

def main():
    global encoder, client, llm, COLLECTION
    args = get_args()
    COLLECTION = args.collection
    encoder = SentenceTransformer("mixedbread-ai/mxbai-embed-large")
    client = QdrantClient(":memory:")
    # expect embed.py already filled the collection
    if not client.collection_exists(COLLECTION):
        raise SystemExit(f"Collection '{COLLECTION}' not found, run rag/embed.py first.")
    llm = LLM(model=args.ckpt, gpu_memory_utilization=args.gpu_memory_utilization)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
