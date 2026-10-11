import hashlib
import os
import uuid
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from .config import COLLECTION, DATA, MODEL, EMBEDDING_ENGINE, VECTOR_COLLECTIONS, TEXT_FIELDS
from .store import tokens
from .embeddings import CloudflareEmbeddings

class HybridRetriever:
    def __init__(self, store, embedder=None):
        self.store = store
        self.verses = store.all()
        if not self.verses:
            raise RuntimeError("Dataset missing. Run python -m quran_rag.ingest")
        self.bm25 = BM25Okapi([tokens(v["english_translation"]) for v in self.verses])
        url = os.getenv("QDRANT_URL")
        self.client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY") or None) if url else QdrantClient(path=str(DATA / "qdrant"))
        self._embedder = embedder
        self._compatible_counts = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = CloudflareEmbeddings()
        return self._embedder

    def fingerprint(self, verse, language="english"):
        # Preserve the original English fingerprints and collection to reuse work.
        text = verse[TEXT_FIELDS[language]]
        return hashlib.sha256((MODEL + "\n" + EMBEDDING_ENGINE + "\n" + text).encode()).hexdigest()

    def encode_passages(self, batch, language="english"):
        return self.embedder.embed_documents([v[TEXT_FIELDS[language]] for v in batch])

    def index(self, batch_size=32):
        self._compatible_counts = None
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        jobs = []
        for language, collection in VECTOR_COLLECTIONS.items():
            existing = {}
            if self.client.collection_exists(collection):
                offset = None
                while True:
                    points, offset = self.client.scroll(collection, limit=256, offset=offset, with_payload=True, with_vectors=False)
                    existing.update({p.payload["verse_key"]: p.payload.get("fingerprint") if p.payload.get("embedding_engine") == EMBEDDING_ENGINE else None for p in points})
                    if offset is None:
                        break
            pending = [v for v in self.verses if existing.get(v["verse_key"]) != self.fingerprint(v, language)]
            # Similar lengths reduce CPU work spent padding batches.
            pending.sort(key=lambda v: len(v[TEXT_FIELDS[language]]))
            reused = len(self.verses) - len(pending)
            print(f"{language}: reusing {reused}/{len(self.verses)} vectors; {len(pending)} remaining", flush=True)
            jobs.append((language, collection, pending, reused))
        # Alternate languages so both indexes make progress from the outset.
        for start in range(0, max(len(job[2]) for job in jobs), batch_size):
            for language, collection, pending, reused in jobs:
                batch = pending[start:start+batch_size]
                if not batch:
                    continue
                vectors = self.encode_passages(batch, language)
                if not self.client.collection_exists(collection):
                    self.client.create_collection(collection, vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE))
                self.client.upsert(collection, points=[models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, "tanzil:" + v["verse_key"])), vector=vector,
                    payload={"verse_key": v["verse_key"], "surah_number": v["surah_number"], "fingerprint": self.fingerprint(v, language), "embedding_model": MODEL, "embedding_engine": EMBEDDING_ENGINE, "language": language}
                ) for v, vector in zip(batch, vectors)], wait=True)
                print(f"{language}: indexed {reused+min(start+batch_size, len(pending))}/{len(self.verses)} ayahs", flush=True)
        print("Bilingual vector index complete", flush=True)
        self._compatible_counts = None

    def progress(self):
        valid = models.Filter(must=[models.FieldCondition(key="embedding_engine", match=models.MatchValue(value=EMBEDDING_ENGINE))])
        engines = {language: self.client.count(collection, count_filter=valid, exact=True).count if self.client.collection_exists(collection) else 0 for language, collection in VECTOR_COLLECTIONS.items()}
        if self._compatible_counts is None:
            self._compatible_counts = {}
            for language, collection in VECTOR_COLLECTIONS.items():
                expected = {v['verse_key']: self.fingerprint(v, language) for v in self.verses}
                count, offset = 0, None
                if self.client.collection_exists(collection):
                    while True:
                        points, offset = self.client.scroll(collection, limit=256, offset=offset, with_vectors=False, with_payload=True)
                        count += sum(p.payload.get('embedding_engine') == EMBEDDING_ENGINE and p.payload.get('fingerprint') == expected.get(p.payload.get('verse_key')) for p in points)
                        if offset is None:
                            break
                self._compatible_counts[language] = count
        return {language: min(count, self._compatible_counts[language]) for language, count in engines.items()}

    def ready(self):
        return all(count == len(self.verses) for count in self.progress().values())

    def search(self, query, k=12, candidates=50, surah=None):
        if not self.ready():
            raise RuntimeError("Bilingual vector index incomplete. Run python -m quran_rag.ingest")
        scores = self.bm25.get_scores(tokens(query))
        eligible = [i for i, verse in enumerate(self.verses) if surah is None or verse["surah_number"] == surah]
        lexical = [self.verses[i]["verse_key"] for i in sorted(eligible, key=lambda i: scores[i], reverse=True)[:candidates] if scores[i] > 0]
        vector = self.embedder.embed_query(query)
        chapter_filter = models.Filter(must=[models.FieldCondition(key="surah_number", match=models.MatchValue(value=surah))]) if surah is not None else None
        rankings = [lexical]
        for collection in VECTOR_COLLECTIONS.values():
            rankings.append([p.payload["verse_key"] for p in self.client.query_points(collection, query=vector, query_filter=chapter_filter, limit=candidates).points])
        fused = {}
        for ranking in rankings:
            for rank, key in enumerate(ranking, 1):
                fused[key] = fused.get(key, 0) + 1 / (60 + rank)
        ranked = sorted(fused, key=lambda key: (-fused[key], key))[:k]
        return [self.store.get(key) for key in ranked]

    def close(self):
        self.client.close()
        if self._embedder and hasattr(self._embedder, "close"):
            self._embedder.close()
