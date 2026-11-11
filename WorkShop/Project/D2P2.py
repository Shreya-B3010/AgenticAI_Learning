import os
import json
import pickle
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Abstract Base Class (the "protocol" / unified interface)
# ══════════════════════════════════════════════════════════════════════════════

class VectorStore(ABC):
    """
    Abstract base class defining the unified interface all three backends implement.
    
    By programming to this interface instead of to Chroma/FAISS/Pinecone directly,
    the rest of the codebase (ingest.py, rag_engine.py, agents) never needs to change
    when you swap backends. This is the Dependency Inversion principle in practice.
    """

    @abstractmethod
    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        """Add or update documents. Implementations must be idempotent (upsert semantics)."""

    @abstractmethod
    def query(
        self,
        embedding: List[float],
        k: int = 5,
        filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return top-k results for the given query embedding.
        Each result dict must contain: 'id', 'document', 'metadata', 'score'.
        score is cosine similarity (higher = more relevant).
        filter is a backend-specific metadata filter dict (may be ignored by FAISS).
        """

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Remove documents by ID."""

    @abstractmethod
    def persist(self) -> None:
        """Save state to disk (no-op for backends that persist automatically)."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored documents."""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FAISS Backend
# ══════════════════════════════════════════════════════════════════════════════

import faiss  # faiss-cpu package — pure Python import, no server needed


class FaissStore(VectorStore):
    """
    FAISS-backed VectorStore.
    
    What FAISS provides: blazing-fast vector similarity search.
    What FAISS does NOT provide: metadata, document text, persistence, or IDs.
    
    We compensate with:
    - A Python list (self._id_map) mapping FAISS's sequential integer index → our string ID
    - A Python dict (self._sidecar) mapping string ID → {document, metadata}
    - pickle for disk persistence of both sidecar structures
    - Post-query metadata filtering (filter applied AFTER retrieval — less efficient but necessary)
    """

    def __init__(self, dim: int = 768, index_path: str = "./faiss_index"):
        self.dim = dim
        self.index_path = index_path
        self._sidecar_path = index_path + "_sidecar.pkl"

        # IndexFlatIP: exact inner-product search.
        # For cosine similarity, vectors must be L2-normalised before adding.
        # After normalisation, inner product equals cosine similarity.
        self._index = faiss.IndexFlatIP(dim)

        # _id_map[faiss_int_idx] = our string ID — grows as we add documents.
        # FAISS assigns sequential integers 0, 1, 2, ... internally.
        self._id_map: List[str] = []

        # _sidecar maps our string ID → dict with 'document' and 'metadata'.
        self._sidecar: Dict[str, Dict] = {}

        # Load from disk if an existing index is found.
        self._load_if_exists()

    def _load_if_exists(self):
        """Restore index + sidecar from disk if they exist from a prior run."""
        if os.path.exists(self.index_path + ".index"):
            self._index = faiss.read_index(self.index_path + ".index")
            with open(self._sidecar_path, "rb") as f:
                saved = pickle.load(f)
                self._id_map = saved["id_map"]
                self._sidecar = saved["sidecar"]
            print(f"[FAISS] Loaded {self._index.ntotal} vectors from {self.index_path}")

    def _normalise(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalise rows so that inner product = cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)   # avoid division by zero for zero vectors
        return vectors / norms

    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        """
        Add documents. FAISS does not support true upserts — we simulate it by
        skipping IDs that already exist (idempotent but not update-aware).
        For a true update, delete first then re-add.
        """
        new_ids, new_docs, new_metas, new_embs = [], [], [], []
        for id_, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
            if id_ not in self._sidecar:   # skip existing IDs
                new_ids.append(id_)
                new_docs.append(doc)
                new_metas.append(meta)
                new_embs.append(emb)

        if not new_ids:
            return   # nothing to add

        vectors = np.array(new_embs, dtype=np.float32)
        vectors = self._normalise(vectors)
        self._index.add(vectors)    # FAISS assigns sequential int IDs internally

        for id_, doc, meta in zip(new_ids, new_docs, new_metas):
            self._id_map.append(id_)
            self._sidecar[id_] = {"document": doc, "metadata": meta}

    def query(
        self,
        embedding: List[float],
        k: int = 5,
        filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for top-k nearest neighbours.
        filter is applied POST-retrieval (FAISS has no native metadata filtering).
        We over-fetch (k*10) to ensure enough results survive filtering.
        """
        if self._index.ntotal == 0:
            return []

        vec = np.array([embedding], dtype=np.float32)
        vec = self._normalise(vec)

        # Over-fetch to have headroom for post-filter elimination.
        fetch_k = min(k * 10, self._index.ntotal)
        scores, faiss_indices = self._index.search(vec, fetch_k)
        # scores[0] are inner-product scores (= cosine similarity after normalisation)
        # faiss_indices[0] are the internal sequential integer IDs

        results = []
        for score, idx in zip(scores[0], faiss_indices[0]):
            if idx == -1:               # FAISS returns -1 for "no result found"
                continue
            if idx >= len(self._id_map):
                continue                # guard against stale index

            str_id = self._id_map[idx]
            if str_id not in self._sidecar:
                continue

            meta = self._sidecar[str_id]["metadata"]

            # Apply metadata filter if provided (simple equality check)
            if filter:
                match = all(meta.get(key) == val for key, val in filter.items())
                if not match:
                    continue

            results.append({
                "id":       str_id,
                "document": self._sidecar[str_id]["document"],
                "metadata": meta,
                "score":    float(score),
            })

            if len(results) == k:
                break

        return results

    def delete(self, ids: List[str]) -> None:
        """
        Mark IDs as deleted in the sidecar — FAISS IndexFlatIP does not support
        in-place deletion. Deleted entries are excluded from future query results
        but remain in the FAISS index (memory leak for large churn workloads;
        rebuild the index periodically in production).
        """
        for id_ in ids:
            if id_ in self._sidecar:
                del self._sidecar[id_]

    def persist(self) -> None:
        """Save the FAISS index and sidecar structures to disk."""
        faiss.write_index(self._index, self.index_path + ".index")
        with open(self._sidecar_path, "wb") as f:
            pickle.dump({"id_map": self._id_map, "sidecar": self._sidecar}, f)
        print(f"[FAISS] Persisted {self._index.ntotal} vectors → {self.index_path}")

    def count(self) -> int:
        """Return number of live (non-deleted) documents."""
        return len(self._sidecar)


from pinecone import Pinecone, ServerlessSpec


class PineconeStore(VectorStore):
    """
    Pinecone-backed VectorStore (serverless free tier).
    
    Pinecone provides everything FAISS lacks: persistence, metadata, native filtering, and scaling.
    The free tier supports one serverless index. Keep index names short and consistent.
    
    One important constraint: the index dimension must match your embedding model's output size
    exactly. A mismatch causes a 400 error on upsert — it's the most common student error here.
    Always verify: EMBED_DIM matches what your model actually outputs.
    """

    def __init__(
        self,
        index_name: str = "day2-demo",
        dim: int = 768,
        namespace: str = "default",
    ):
        self.namespace = namespace
        self.dim = dim

        # Initialise Pinecone client — reads PINECONE_API_KEY from environment.
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        # Create the index if it doesn't exist — idempotent.
        # ServerlessSpec(cloud, region) selects the free-tier serverless offering.
        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=dim,
                metric="cosine",   # cosine similarity — aligns with our embedding choice
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"[Pinecone] Created index '{index_name}' (dim={dim}, metric=cosine)")
        else:
            print(f"[Pinecone] Using existing index '{index_name}'")

        self._index = pc.Index(index_name)

    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        """
        Upsert vectors into Pinecone. Pinecone natively supports upsert semantics.
        
        We store the document text inside the metadata under the key '_document'
        because Pinecone doesn't have a separate document field — metadata is the payload.
        This is a common Pinecone pattern.
        
        Batching: Pinecone recommends batch sizes of 100 for free-tier upserts.
        """
        vectors = []
        for id_, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
            pinecone_meta = dict(meta)          # copy to avoid mutating caller's dict
            pinecone_meta["_document"] = doc    # store text alongside metadata
            vectors.append({
                "id":     id_,
                "values": emb,
                "metadata": pinecone_meta,
            })

        # Batch upserts in groups of 100
        BATCH_SIZE = 100
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i : i + BATCH_SIZE]
            self._index.upsert(vectors=batch, namespace=self.namespace)

        print(f"[Pinecone] Upserted {len(vectors)} vectors → namespace='{self.namespace}'")

    def query(
        self,
        embedding: List[float],
        k: int = 5,
        filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query Pinecone for top-k nearest neighbours.
        filter is passed directly to Pinecone's native metadata filter (very efficient).
        Pinecone filter syntax: {"key": {"$eq": value}} — same operators as Chroma.
        """
        resp = self._index.query(
            vector=embedding,
            top_k=k,
            namespace=self.namespace,
            filter=filter,
            include_metadata=True,   # always include metadata so we can return the document text
        )

        results = []
        for match in resp.matches:
            meta = dict(match.metadata)
            doc = meta.pop("_document", "")   # extract document text we stored earlier
            results.append({
                "id":       match.id,
                "document": doc,
                "metadata": meta,
                "score":    float(match.score),   # cosine similarity score from Pinecone
            })

        return results

    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID from the current namespace."""
        self._index.delete(ids=ids, namespace=self.namespace)

    def persist(self) -> None:
        """Pinecone persists automatically — no explicit action needed."""
        print("[Pinecone] No explicit persist needed — Pinecone is always persisted.")

    def count(self) -> int:
        """Return the total vector count in the current namespace."""
        stats = self._index.describe_index_stats()
        ns_stats = stats.namespaces.get(self.namespace)
        return ns_stats.vector_count if ns_stats else 0

import chromadb


class ChromaStore(VectorStore):
    """
    Chroma-backed VectorStore. Wraps the Chroma API we learned in Session 1
    behind the unified interface.
    
    Chroma natively handles: persistence, metadata filters, document text, IDs.
    Our job is just to translate the unified interface calls to Chroma API calls.
    """

    def __init__(self, collection_name: str = "default", db_path: str = "./chroma_db"):
        self._client = chromadb.PersistentClient(path=db_path)
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        self._col.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(
        self,
        embedding: List[float],
        k: int = 5,
        filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        resp = self._col.query(
            query_embeddings=[embedding],
            n_results=k,
            where=filter,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for doc, meta, dist in zip(
            resp["documents"][0],
            resp["metadatas"][0],
            resp["distances"][0],
        ):
            results.append({
                "id":       "",              # Chroma returns IDs separately; omit for simplicity
                "document": doc,
                "metadata": meta,
                "score":    1 - dist,        # cosine distance → cosine similarity
            })

        return results

    def delete(self, ids: List[str]) -> None:
        self._col.delete(ids=ids)

    def persist(self) -> None:
        """Chroma PersistentClient auto-persists on write — this is a no-op."""
        pass

    def count(self) -> int:
        return self._col.count()


# ── Factory function ──────────────────────────────────────────────────────────

def get_vector_store(
    backend: str,
    collection_name: str = "default",
    dim: int = 768,
    db_path: str = "./chroma_db",
    pinecone_namespace: str = "default",
) -> VectorStore:
    """
    Factory: return the correct VectorStore implementation by backend name.
    
    Usage:
        store = get_vector_store("chroma")
        store = get_vector_store("faiss", dim=768)
        store = get_vector_store("pinecone", dim=768)
    
    This is the single call site every other module uses — swap '--backend chroma' to
    '--backend faiss' at the CLI and the entire pipeline uses a different engine.
    """
    backend = backend.lower().strip()
    if backend == "chroma":
        return ChromaStore(collection_name=collection_name, db_path=db_path)
    elif backend == "faiss":
        return FaissStore(dim=dim, index_path=f"./{collection_name}_faiss")
    elif backend == "pinecone":
        return PineconeStore(
            index_name=collection_name.replace("_", "-"),   # Pinecone names: lowercase, hyphens only
            dim=dim,
            namespace=pinecone_namespace,
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose: chroma | faiss | pinecone")

    # ── Execution Block ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Initializing Vector Store Engine...")
    
    # 1. Use the factory function to initialize a local FAISS store
    # This automatically invokes __init__() and looks for existing indexes
    store = get_vector_store(backend="faiss", collection_name="demo_collection", dim=768)
    
    # 2. Prepare sample mock data matching the protocol's required structures
    sample_ids = ["id_01", "id_02"]
    sample_docs = [
        "Dependency Inversion allows you to swap database backends seamlessly.",
        "FAISS focuses purely on fast similarity vector operations."
    ]
    sample_metas = [{"topic": "architecture"}, {"topic": "performance"}]
    
    # Generate dummy vectors (each vector must match the 768 dimensions specified)
    sample_embs = [
        [0.01] * 768,
        [0.05] * 768
    ]
    
    # 3. Insert the vectors to execute the internal logic
    print("\n[Action] Writing sample documents into the FAISS store...")
    store.add(
        ids=sample_ids,
        documents=sample_docs,
        metadatas=sample_metas,
        embeddings=sample_embs
    )
    
    # 4. Read back the updated item count
    print(f"[Status] Total active documents in store: {store.count()}")
    
    # 5. Manually trigger disk saving to print your persistence notification
    print("\n[Action] Persisting state to local files...")
    store.persist()