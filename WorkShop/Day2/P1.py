# Import OS utilities for environment variables.
import os

# Import JSON utilities (currently unused in this snippet).
import json

# Import time utilities (currently unused in this snippet).
import time

# Import type hints for lists.
from typing import List

# Load environment variables from a .env file.
from dotenv import load_dotenv

# Import the Google GenAI SDK client.
from google import genai

# Import the Ollama Python client.
import ollama


# Load .env variables into the environment.
load_dotenv()


# Read the Gemini API key from the environment.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Create one reusable Gemini client.
client = genai.Client(api_key=GEMINI_API_KEY)


# ── Embedding model constants ────────────────────────────────────────────────
# gemini-embedding is the free-tier embedding model — check AI Studio for current name.
# nomic-embed-text produces 768-dimensional vectors; gemini-embedding produces 768 as well
# (verify in AI Studio console — the dimension must match your Pinecone index on Day 2 S2).

# Gemini embedding model name.
GEMINI_EMBED_MODEL  = "gemini-embedding-exp-03-07"   # free-tier; check AI Studio for latest name

# Local Ollama embedding model name.
OLLAMA_EMBED_MODEL  = "nomic-embed-text"              # 768-dim; fast; fully offline

# Expected vector dimension for both models.
EMBED_DIM           = 768                             # shared dimension for both models



def embed_gemini(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts using the Gemini free-tier embedding model.
    Returns a list of float vectors (one per input text).
    
    The task_type parameter hints to the model how the embedding will be used:
    - RETRIEVAL_DOCUMENT  → for chunks being stored in the vector DB
    - RETRIEVAL_QUERY     → for the user's question at query time
    Using the correct task_type improves retrieval quality.
    """

    # List to collect one vector per input text.
    embeddings = []

    # Embed each text individually.
    for text in texts:
        # Request a document embedding from Gemini.
        resp = client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=text,
            config={"task_type": "RETRIEVAL_DOCUMENT"},
        )

        # resp.embeddings is a list of ContentEmbedding objects — take the first (one text in).
        # Store the numeric vector for this text.
        embeddings.append(resp.embeddings[0].values)

    # Return all document vectors.
    return embeddings



def embed_query_gemini(query: str) -> List[float]:
    """Embed a single query string with task_type=RETRIEVAL_QUERY for retrieval."""

    # Request a query embedding from Gemini.
    resp = client.models.embed_content(
        model=GEMINI_EMBED_MODEL,
        contents=query,
        config={"task_type": "RETRIEVAL_QUERY"},
    )

    # Return the query's vector.
    return resp.embeddings[0].values



def embed_ollama(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts using nomic-embed-text via Ollama.
    Fully offline — no API key, no rate limits.
    Ollama's embed endpoint accepts a single prompt at a time; we loop.
    """

    # List to collect one vector per input text.
    embeddings = []

    # Embed each text individually.
    for text in texts:
        # Request an embedding from the local Ollama model.
        resp = ollama.embed(model=OLLAMA_EMBED_MODEL, input=text)

        # resp["embeddings"] is a list with one entry (we passed one text).
        # Store the numeric vector for this text.
        embeddings.append(resp["embeddings"][0])

    # Return all document vectors.
    return embeddings



def embed_query_ollama(query: str) -> List[float]:
    """Embed a single query string via Ollama nomic-embed-text."""

    # Request a query embedding from Ollama.
    resp = ollama.embed(model=OLLAMA_EMBED_MODEL, input=query)

    # Return the query's vector.
    return resp["embeddings"][0]



# Confirm that both embedding paths are configured.
print("[Setup] Embedding helpers ready — Gemini + Ollama paths defined.")

# Print the Gemini embedding model.
print(f"        Gemini model : {GEMINI_EMBED_MODEL}")

# Print the Ollama embedding model.
print(f"        Ollama model : {OLLAMA_EMBED_MODEL}")

# Print the expected embedding dimension.
print(f"        Expected dim : {EMBED_DIM}")

# Chroma DB
# Import the Chroma vector database client.
import chromadb


# PersistentClient stores data on disk at the given path.
# On subsequent runs, it LOADS existing collections — no need to re-add documents.
# Use this for the capstone corpus (you want it to survive restarts).

# Create a disk-backed Chroma client using ./chroma_db.
chroma_persistent = chromadb.PersistentClient(path="./chroma_db")


# EphemeralClient stores data only in RAM — wiped when the process exits.
# Useful for one-off experiments or unit tests where you don't want leftovers.

# Create an in-memory Chroma client.
chroma_ephemeral = chromadb.EphemeralClient()


# Confirm the persistent client's storage location.
print(f"\n[Chroma] Persistent client ready → ./chroma_db")

# Confirm the ephemeral client is memory-only.
print(f"[Chroma] Ephemeral client ready  → in-memory only")

# List names of collections already saved in the persistent database.
print(f"[Chroma] Existing collections    : {[c.name for c in chroma_persistent.list_collections()]}")

# Name of the Chroma collection to use.
COLLECTION_NAME = "day2_demo"


# get_or_create_collection is idempotent — safe to call on every run.
# If the collection already exists, it returns it. If not, it creates it.
# embedding_function=None means WE supply embeddings manually on every add/query call.
# This is critical for model portability — the collection isn't locked to one embed model.

# Get the existing collection or create it if missing.
collection = chroma_persistent.get_or_create_collection(
    name=COLLECTION_NAME,

    # Use cosine distance instead of the default L2 distance.
    metadata={"hnsw:space": "cosine"},  # use cosine distance (vs default L2) — better for text
)


# ── Sample documents with rich metadata ──────────────────────────────────────
# Metadata is a flat dict (string/int/float/bool values only — no nested dicts).
# Think of it as the "columns" of a schema — enables filtered retrieval.

# Text chunks to embed and store.
documents = [
    "Chroma uses HNSW for approximate nearest neighbour search over dense embeddings.",
    "FAISS is a C++ library by Meta AI for efficient similarity search at billion-vector scale.",
    "Pinecone provides a fully managed vector database service with serverless scaling.",
    "RAG (Retrieval-Augmented Generation) grounds LLM answers in retrieved documents, reducing hallucination.",
    "Chunking splits documents into segments before embedding — chunk size is the #1 RAG quality lever.",
    "Cosine similarity measures the angle between two vectors; magnitude is ignored.",
    "nomic-embed-text is an open-source embedding model that runs locally via Ollama.",
    "Exponential backoff with jitter prevents thundering-herd retries on 429 rate-limit responses.",
]


# Metadata attached to each document for filtering and provenance.
metadatas = [
    {"source": "chroma_docs",   "topic": "vector_db",   "section": "architecture", "day": 2},
    {"source": "faiss_docs",    "topic": "vector_db",   "section": "library",      "day": 2},
    {"source": "pinecone_docs", "topic": "vector_db",   "section": "managed",      "day": 2},
    {"source": "rag_overview",  "topic": "rag",         "section": "concepts",     "day": 3},
    {"source": "rag_overview",  "topic": "chunking",    "section": "strategy",     "day": 2},
    {"source": "math_primer",   "topic": "embeddings",  "section": "similarity",   "day": 1},
    {"source": "ollama_docs",   "topic": "embeddings",  "section": "local_model",  "day": 2},
    {"source": "api_patterns",  "topic": "engineering", "section": "reliability",  "day": 1},
]


# Generate a unique string ID for every document.
ids = [f"doc_{i}" for i in range(len(documents))]


# ── Generate embeddings (Ollama path — no API rate limits for this demo) ──
# Announce embedding generation.
print("\n[Chroma] Generating embeddings via nomic-embed-text (Ollama)...")

# Start timer.
t0 = time.perf_counter()

# Create one embedding vector per document using local Ollama.
embeddings = embed_ollama(documents)

# Print elapsed time and vector dimension.
print(f"         Done in {time.perf_counter()-t0:.2f}s | dim={len(embeddings[0])}")


# ── Add to collection ──────────────────────────────────────────────────────
# add() is idempotent if IDs already exist — it will RAISE an error on duplicate IDs.
# Use upsert() if you want add-or-update semantics (safer for re-runs).

# Store documents, metadata, IDs, and precomputed embeddings in Chroma.
collection.upsert(
    ids=ids,
    documents=documents,   # original text — stored alongside the embedding for retrieval
    metadatas=metadatas,
    embeddings=embeddings, # our manually computed vectors — Chroma stores them as-is
)


# Count how many documents are now in the collection.
count = collection.count()

# Print the collection size.
print(f"[Chroma] Collection '{COLLECTION_NAME}': {count} documents")

# Preview the first two stored documents.
print(f"[Chroma] Peek (first 2): {collection.peek(limit=2)['documents']}")

def query_chroma(query: str, n_results: int = 3, where: dict = None) -> None:
    """
    Query the Chroma collection and print results with distances.
    
    n_results: how many nearest neighbours to return (top-k).
    where: optional metadata filter dict — only documents matching ALL conditions are candidates.
    distances: cosine distance (0=identical, 2=opposite); lower is more relevant.
    """

    # Embed the query using the same Ollama model used for stored documents.
    query_embedding = embed_query_ollama(query)  # embed the query with the same model as the docs


    # Search Chroma for the closest stored documents.
    results = collection.query(
        query_embeddings=[query_embedding],   # list of query vectors (one here)
        n_results=n_results,                  # return top-k matches
        where=where,                          # None means no filter — search all documents
        include=["documents", "metadatas", "distances"],
    )


    # Print the query and any active metadata filter.
    print(f"\n── Query: '{query}' | filter={where} ──")

    # Loop through each returned document, metadata record, and distance.
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        # Convert cosine distance to cosine similarity.
        relevance = 1 - dist   # cosine similarity = 1 - cosine distance

        # Print rank, relevance score, and metadata.
        print(f"  [{i+1}] score={relevance:.3f} | source={meta['source']} | topic={meta['topic']}")

        # Print the first 100 characters of the matched document.
        print(f"       {doc[:100]}...")



# ── Run several queries: with and without metadata filters ───────────────────


# No filter — search all 8 documents
query_chroma("How does approximate nearest neighbour search work?", n_results=3)


# Metadata filter: only documents about vector databases
query_chroma("What are the options for storing vectors?", n_results=3,
             where={"topic": {"$eq": "vector_db"}})


# Metadata filter: only Day 1 content
query_chroma("How to handle API rate limits?", n_results=2,
             where={"day": {"$eq": 1}})


# Metadata filter using $in (match any of several values)
query_chroma("How are embeddings used in RAG?", n_results=3,
             where={"topic": {"$in": ["rag", "embeddings"]}})


# Update a document — upsert() handles both add-new and update-existing.

# New replacement text for document doc_0.
updated_text = "Chroma uses HNSW for ANN search. It supports cosine, L2, and IP distance metrics."

# Generate a fresh embedding for the updated text.
updated_embedding = embed_ollama([updated_text])[0]


# Replace the existing doc_0 record using its same ID.
collection.upsert(
    ids=["doc_0"],                          # same ID as the original doc_0
    documents=[updated_text],
    metadatas=[{"source": "chroma_docs", "topic": "vector_db", "section": "architecture", "day": 2}],
    embeddings=[updated_embedding],
)

# Fetch and preview the updated document text.
print(f"\n[Chroma] Updated doc_0. New text: {collection.get(ids=['doc_0'])['documents'][0][:80]}")


# Delete a document by ID
collection.delete(ids=["doc_7"])   # remove the backoff doc

# Print the new total document count after deletion.
print(f"[Chroma] Deleted doc_7. Collection now has {collection.count()} documents.")

# Name of the persistent Chroma collection for the capstone corpus.
CAPSTONE_COLLECTION = "capstone_corpus"


# Get the capstone collection, or create it if it does not exist.
capstone_col = chroma_persistent.get_or_create_collection(
    name=CAPSTONE_COLLECTION,

    # Use cosine distance for text similarity search.
    metadata={"hnsw:space": "cosine"},
)


# Sample capstone snippets — replace these with actual corpus chunks in Session 3.
# Each represents a chunk from a hypothetical "Campus Knowledge Base" (college handbook).

# Text chunks representing campus handbook content.
capstone_chunks = [
    "The library is open Monday to Saturday from 8 AM to 8 PM. Students must carry their ID cards to borrow books.",
    "A student may borrow up to 3 books at a time. Books are due in 14 days. Late returns incur a fine of ₹5 per day per book.",
    "The computer lab in Block C has 60 workstations. It is open 24 hours during examination periods.",
    "Fee payment for even semesters is due by the 15th of January. Late payment attracts a penalty of ₹500.",
    "The Anti-Ragging Committee meets on the first Monday of every month. Report incidents to the Dean of Students.",
    "Attendance below 75% in any subject results in automatic detention. Medical certificates must be submitted within 5 days.",
    "The sports complex includes a basketball court, two badminton courts, and a 400m running track.",
    "Industry internships of at least 4 weeks are mandatory for all 6th-semester students. Completion must be certified by the company.",
    "The research lab on the 3rd floor of the CSE department is equipped with GPU workstations for deep learning projects.",
    "Students must register for electives by the end of the 2nd week of each semester via the online portal.",
]


# Metadata describing each chunk's source, page, section, and topic.
capstone_meta = [
    {"source": "college_handbook.pdf", "page": 12, "section": "library", "topic": "rules"},
    {"source": "college_handbook.pdf", "page": 13, "section": "library", "topic": "fines"},
    {"source": "college_handbook.pdf", "page": 20, "section": "facilities", "topic": "labs"},
    {"source": "college_handbook.pdf", "page": 5,  "section": "fees",      "topic": "payment"},
    {"source": "college_handbook.pdf", "page": 8,  "section": "conduct",   "topic": "anti_ragging"},
    {"source": "college_handbook.pdf", "page": 9,  "section": "academics", "topic": "attendance"},
    {"source": "college_handbook.pdf", "page": 22, "section": "facilities","topic": "sports"},
    {"source": "college_handbook.pdf", "page": 30, "section": "academics", "topic": "internship"},
    {"source": "college_handbook.pdf", "page": 21, "section": "facilities","topic": "research"},
    {"source": "college_handbook.pdf", "page": 11, "section": "academics", "topic": "registration"},
]


# Create unique IDs for all capstone chunks.
capstone_ids = [f"cap_{i}" for i in range(len(capstone_chunks))]


# Announce local embedding generation.
print("\n[Capstone] Embedding corpus chunks via nomic-embed-text...")

# Generate one embedding vector per corpus chunk.
cap_embeddings = embed_ollama(capstone_chunks)


# Store chunks, metadata, IDs, and vectors in the capstone collection.
capstone_col.upsert(
    ids=capstone_ids,
    documents=capstone_chunks,
    metadatas=capstone_meta,
    embeddings=cap_embeddings,
)


# Confirm how many chunks were ingested.
print(f"[Capstone] Ingested {capstone_col.count()} chunks into '{CAPSTONE_COLLECTION}'.")


# Smoke query — verify retrieval works on the capstone corpus

# Embed the test question.
cap_q = embed_query_ollama("What is the fine for late book return?")

# Retrieve the three most similar handbook chunks.
cap_results = capstone_col.query(
    query_embeddings=[cap_q],
    n_results=3,
    include=["documents", "metadatas", "distances"],
)

# Print the smoke-query header.
print("\n[Capstone Smoke Query] 'What is the fine for late book return?'")

# Display each retrieved chunk with relevance and source details.
for doc, meta, dist in zip(cap_results["documents"][0], cap_results["metadatas"][0], cap_results["distances"][0]):
    print(
    f"  score={1-dist:.3f} | "
    f"{meta.get('source', 'Unknown')} "
    f"p.{meta.get('page', 'N/A')} "
    f"[{meta.get('section', 'Unspecified')}]"
    )
    print(f"  {doc[:110]}")