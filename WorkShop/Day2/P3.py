import os
import re
import chromadb
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# ── Synthetic sample document  ──────────
# Students use their actual PDF — this synthetic text lets the experiment run without a PDF.
SAMPLE_TEXT = """
Chapter 1: Library Rules and Regulations

The Central Library at the institute is one of its most valuable resources. It operates from 
Monday to Saturday, 8 AM to 8 PM. During examination periods it remains open until 10 PM.
Students must present a valid ID card to enter the library premises.

Borrowing Policy
Students in undergraduate programs may borrow up to 3 books simultaneously. Post-graduate 
students may borrow up to 5 books. Books are issued for a period of 14 days. Books may be 
renewed once for an additional 14 days, provided no other student has placed a request.

Fines and Penalties
Late returns incur a fine of rupees 5 per day per book. If a book is lost, the student must 
replace it with a new copy of the same edition or pay three times the listed price. Damage 
to library property, including highlighting or writing in books, results in a minimum fine 
of rupees 200 per incident.

Reference Section
The reference section on the second floor houses encyclopedias, journals, and rare collections. 
Books in this section cannot be borrowed and must be read within the library. Photocopying of 
reference material is permitted to a maximum of 20 pages per visit.

Chapter 2: Laboratory Access and Usage

The Computer Science department operates five laboratories. Labs 1, 2, and 3 are general 
purpose labs with 60 workstations each. Lab 4 is the networking lab. Lab 5 is the AI/ML 
research lab with GPU-equipped workstations.

Access Hours
Regular labs are accessible during class hours (9 AM to 5 PM on working days). The AI/ML 
lab (Lab 5) requires prior booking via the departmental coordinator. During project season 
(November to April) all labs remain open until 9 PM on working days.

Equipment Booking
External drives must not be connected without the lab assistant's permission. All print jobs 
must be submitted to the lab assistant. Color printing is charged at rupees 5 per page.

Chapter 3: Fee Payment Procedures

Semester fees must be paid before the deadline specified in the academic calendar. For odd 
semesters the payment deadline is August 15. For even semesters the deadline is January 15. 

Late Payment
A late fee of rupees 500 is charged for payments made after the deadline. Students with 
outstanding dues are not permitted to appear for end-semester examinations. Fee concession 
applications must be submitted to the accounts office by July 31 for odd semesters.

Scholarship Disbursement
Government scholarship amounts are directly credited to the student's bank account. The 
institute does not handle scholarship funds. Students must ensure their bank account details 
are updated in the student portal.
"""


# ════════════════════════════════════════════════════════════════════════════
# CHUNKING STRATEGY IMPLEMENTATIONS
# ════════════════════════════════════════════════════════════════════════════

def chunk_fixed_size(text: str, size: int = 500, overlap: int = 0) -> List[str]:
    """
    Fixed-size character chunking with optional overlap.
    Simple but ignores all natural language boundaries.
    A word or sentence may be cut in the middle.
    """
    chunks = []
    step = size - overlap                   # how far to advance the window each time
    if step <= 0:
        raise ValueError("overlap must be less than size")
    for start in range(0, len(text), step):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_recursive(text: str, size: int = 800, overlap: int = 120) -> List[str]:
    """
    Recursive character splitting — the LangChain RecursiveCharacterTextSplitter logic
    implemented from scratch so you understand it without the abstraction.
    
    Algorithm: try to split on the first separator that produces chunks ≤ size.
    Separators tried in order: double newline, single newline, period+space, space, any char.
    This means: prefer paragraph splits → sentence splits → word splits → character splits.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, seps: List[str]) -> List[str]:
        if not seps or len(text) <= size:
            return [text] if text.strip() else []
        sep = seps[0]
        parts = text.split(sep) if sep else list(text)
        results = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    results.append(current)
                # Recurse on a part that is still too long
                if len(part.strip()) > size:
                    results.extend(_split(part.strip(), seps[1:]))
                    current = ""
                else:
                    current = part.strip()
        if current:
            results.append(current)
        return results

    raw_chunks = _split(text, separators)

    # Apply overlap: for each chunk after the first, prepend the tail of the previous chunk.
    if overlap == 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tail = raw_chunks[i-1][-overlap:]   # last `overlap` chars of previous chunk
        overlapped.append((prev_tail + " " + raw_chunks[i]).strip())
    return overlapped


# ── Ingest each strategy into its own Chroma collection ──────────────────────

from WorkShop.Day2.P1 import embed_ollama, embed_query_ollama

chroma_client = chromadb.EphemeralClient()   # ephemeral for the experiment — no leftovers

strategies = [
    ("fixed_500_0",   chunk_fixed_size(SAMPLE_TEXT, size=500,  overlap=0)),
    ("fixed_1000_150",chunk_fixed_size(SAMPLE_TEXT, size=1000, overlap=150)),
    ("recursive_800", chunk_recursive(SAMPLE_TEXT,  size=800,  overlap=120)),
]

print("\n[Chunking] Experiment setup:")
for name, chunks in strategies:
    print(f"  {name:<20}: {len(chunks)} chunks | avg len={sum(len(c) for c in chunks)//len(chunks)} chars")

collections = {}
for name, chunks in strategies:
    col = chroma_client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    embs = embed_ollama(chunks)
    col.add(
        ids=[f"{name}_{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"strategy": name, "chunk_idx": i} for i in range(len(chunks))],
        embeddings=embs,
    )
    collections[name] = col
    print(f"  Ingested '{name}': {col.count()} chunks")

# ── Run 10 test questions against all three ───────────────────────────────────

TEST_QUESTIONS = [
    "What are the library opening hours?",
    "How much is the fine for returning a book late?",
    "Can I renew a borrowed book?",
    "What happens if I lose a library book?",
    "How many books can a UG student borrow?",
    "What is in Lab 5?",
    "How do I book the AI lab?",
    "When is the fee payment deadline for odd semesters?",
    "What is the late fee penalty for fee payment?",
    "How are government scholarships disbursed?",
]

print("\n\n[Chunking] Retrieval quality comparison — top-1 answer per strategy per question")
print("="*100)

votes = {name: 0 for name, _ in strategies}

for q in TEST_QUESTIONS:
    q_emb = embed_query_ollama(q)
    print(f"\nQ: {q}")
    for name, _ in strategies:
        res = collections[name].query(
            query_embeddings=[q_emb], n_results=1,
            include=["documents", "distances"]
        )
        best = res["documents"][0][0] if res["documents"][0] else "(no result)"
        score = 1 - res["distances"][0][0] if res["distances"][0] else 0
        print(f"  [{name:<20}] score={score:.3f} | {best[:90].replace(chr(10), ' ')}...")

# Manual vote: students mark which strategy gave the best top-1 answer for each question.
# Tally determines the "winner" chunking strategy for the capstone.
print("\n[Chunking] Vote for the best answer for each question above.")
print("           Tally in lab2_3_chunking_benchmark.md (template below).")