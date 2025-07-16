"""
Chunk docs, embed with SBERT, and push to Qdrant.
"""
import argparse, json, pathlib, tqdm, uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/big_qa.jsonl")
    p.add_argument("--chunks", type=int, default=512, help="tokens per chunk")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--collection", default="ds_qa_chunks")
    return p.parse_args()

def main():
    args = get_args()
    encoder = SentenceTransformer("mixedbread-ai/mxbai-embed-large")
    client  = QdrantClient(":memory:")
    client.recreate_collection(
        collection_name=args.collection,
        vectors_config=models.VectorParams(size=encoder.get_sentence_embedding_dimension(),
                                           distance=models.Distance.COSINE),
    )
    def stream_jsonl(path):
        with open(path) as f:
            for line in f:
                yield json.loads(line)

    batch, payloads = [], []
    for row in tqdm.tqdm(stream_jsonl(args.input), desc="Embedding"):
        text = f"Q: {row['question']}\nA: {row['answer']}"
        batch.append(text); payloads.append(row)
        if len(batch) == args.batch_size:
            vecs = encoder.encode(batch, show_progress_bar=False)
            ids  = [str(uuid.uuid4()) for _ in vecs]
            client.upsert(args.collection, list(zip(ids, vecs, payloads)))
            batch, payloads = [], []
    if batch:
        vecs = encoder.encode(batch, show_progress_bar=False)
        ids  = [str(uuid.uuid4()) for _ in vecs]
        client.upsert(args.collection, list(zip(ids, vecs, payloads)))
    print("Indexed", client.count(args.collection).count, "chunks")

if __name__ == "__main__":
    main()
