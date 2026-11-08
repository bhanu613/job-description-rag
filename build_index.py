import hashlib
import json
from pathlib import Path
from rag import COLLECTION_NAME, DB_PATH, EMBED_MODEL, ROOT

CHUNK_WORDS = 220
OVERLAP_WORDS = 40
BATCH_SIZE = 8


def build_records(jobs):
    records, seen = [], set()
    for job in jobs:
        if job["source_id"] in seen:
            raise ValueError(f"Duplicate source ID: {job['source_id']}")
        seen.add(job["source_id"])
        words = job["text"].split()
        if not words:
            raise ValueError(f"Empty description: {job['source_id']}")
        for index, start in enumerate(range(0, len(words), CHUNK_WORDS - OVERLAP_WORDS)):
            records.append({
                "id": f"{job['source_id']}-C{index:03d}",
                "text": f"Job title: {job['title']}\n" + " ".join(words[start:start + CHUNK_WORDS]),
                "metadata": {
                    "source_id": job["source_id"], "title": job["title"],
                    "chunk_index": index, "dataset_row": job["dataset_row"],
                    "source_url": job["source_url"], "revision": job["revision"],
                },
            })
            if start + CHUNK_WORDS >= len(words):
                break
    if not records:
        raise ValueError("No records to index; existing collection was not cleared.")
    return records


def sync_collection(collection, records, embed_fn):
    vectors = []
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        embeddings = embed_fn(["search_document: " + r["text"] for r in batch])
        if len(embeddings) != len(batch):
            raise ValueError("Embedding count does not match batch size.")
        vectors.extend(embeddings)
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        collection.upsert(
            ids=[r["id"] for r in batch], documents=[r["text"] for r in batch],
            metadatas=[r["metadata"] for r in batch],
            embeddings=vectors[start:start + len(batch)],
        )
        print(f"Indexed {start + len(batch)}/{len(records)}", flush=True)
    desired = {r["id"] for r in records}
    existing = set(collection.get(include=[])["ids"])
    stale = sorted(existing - desired)
    if stale:
        collection.delete(ids=stale)
    if collection.count() != len(records):
        raise RuntimeError("Stored count does not match the prepared index.")
    return len(stale)


def main():
    import chromadb
    import ollama
    data_path = ROOT / "data" / "jobs.jsonl"
    jobs = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = build_records(jobs)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=None)
    print(f"Descriptions: {len(jobs)}")
    print(f"Chunks to index: {len(records)}")
    deleted = sync_collection(collection, records, lambda texts: ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"])
    metadata = {"source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "descriptions": len(jobs), "chunks": len(records),
                "chunk_words": CHUNK_WORDS, "overlap_words": OVERLAP_WORDS,
                "embedding_model": EMBED_MODEL, "collection": COLLECTION_NAME}
    (ROOT / "data" / "index_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Removed obsolete chunks: {deleted}")
    print(f"Stored chunks: {collection.count()}")


if __name__ == "__main__":
    main()
