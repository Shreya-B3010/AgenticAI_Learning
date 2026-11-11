from typing import List, Dict, Any

RAG_PROMPT_TEMPLATE = """You are a precise knowledge-base assistant answering questions
about a campus handbook. Follow the rules below without exception.

Context blocks:
{context}

Rules:
1. Answer ONLY using facts that appear in the context blocks above.
   Do not use prior knowledge, even if you know the answer.
2. After each factual claim, cite the supporting block number in square brackets,
   e.g. [1] or [2][3]. Every claim needs at least one citation.
3. If the context blocks do not contain the information needed, reply with exactly:
   "I don't know — the answer is not in the knowledge base."
   Do not guess, do not blend in outside facts.
4. Keep the answer to 2-6 sentences unless the question demands more detail.

Question: {question}

Answer:"""


def build_prompt(question: str, results: List[Dict[str, Any]]) -> str:
    """
    Render the template with numbered context blocks.

    Each block carries its provenance (source file + page) INLINE so that:
      - the model can cite precisely ([n] maps to a real, traceable chunk), and
      - your code can map [n] → metadata after generation (citations in Session 2).

    `results` is the list returned by retrieve()/store.query() — each item must
    have "document" and "metadata" keys.
    """
    blocks = []
    for i, r in enumerate(results, start=1):
        meta = r.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        # Single-line provenance header, then the chunk text.
        blocks.append(f"[{i}] (source: {source}, page: {page})\n{r['document']}")

    # Empty retrieval still renders — the model then hits abstention rule #3.
    context_str = "\n\n".join(blocks) if blocks else "(no context blocks retrieved)"
    return RAG_PROMPT_TEMPLATE.format(context=context_str, question=question)