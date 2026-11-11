import os 
import sys
import time
import json

import ollama
from dotenv import load_dotenv #.env
from google import genai
from google.genai import types 

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
OLLAMA_MODEL = "gemma3:4b"

def ask(prompt:str,system: str="",model:str = "gemini-2.5-flash",temp: float=0.7)->str:
    config = types.GenerateContentConfig(
        temperature=temp,
        system_instruction=system if system else None,
    )
    try:
        resp = client.models.generate_content(model=model,
                                              contents=prompt,
                                              config=config)
        return resp.text.strip()
    except Exception as e:
        error_text = str(e).lower()
        if "429" in str(e) or "resource_exhausted" in error_text or "quota" in error_text or "rate limit" in error_text:
            print(f"[Fallback] Gemini quota exhausted. Using Ollama model '{OLLAMA_MODEL}' instead.")
            content = f"{system}\n\n{prompt}" if system else prompt
            resp = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": content}],
            )
            return resp["message"]["content"].strip()
        raise

if __name__ == "__main__":
    # Zero shot prompting
    zero_shot_prompt = "Classify the sentiment of this review as Positive, Negative or Neutral.\n\n Review:'The course materials were well structured but the lab machines were slow.'"
    result = ask(zero_shot_prompt)

    print(f"[Zer0-Shot]{result}")

    # *****************************************************************
    # Prompt with labeled examples to guide sentiment classification (few-shot learning).
    few_shot_prompt = """
    Classify the sentiment of a review as Positive, Negative, or Neutral.


    Examples:
    Review: "Absolutely loved the hands-on labs, learned so much!" → Positive
    Review: "Confusing explanations and the code didn't run." → Negative
    Review: "It was okay, nothing special." → Neutral


    Now classify:
    Review: "The course materials were well-structured but the lab machines were slow."
    →"""

    # Send the prompt using the reusable ask() helper.
    result = ask(few_shot_prompt, temp=0.0)   # temp=0.0 for classification — deterministic

    # Print the model's classification result.
    print(f"[Few-Shot] {result}")
    # The examples anchor the format — the model mirrors the "→ Label" output pattern.

    # *********************************************************************************************
    # ── Technique 3a: Explicit CoT (show the reasoning steps in the prompt) ────
    cot_prompt = """
    A student submitted 3 assignments worth 20 marks each and 2 exams worth 50 marks each.
    They scored 18, 15, 17 on assignments and 42, 38 on exams.
    What is their total percentage?

    Let's think step by step:
    1. Calculate total marks obtained.
    2. Calculate maximum possible marks.
    3. Compute the percentage.
    """

    result = ask(cot_prompt, temp=0.0)
    print(f"[Explicit CoT]\n{result}\n")

    # ── Technique 3b: Zero-Shot CoT — just add "think step by step" ────────────
    zero_shot_cot_prompt = """
    A student submitted 3 assignments worth 20 marks each and 2 exams worth 50 marks each.
    They scored 18, 15, 17 on assignments and 42, 38 on exams.
    What is their total percentage?

    Think step by step before answering.
    """

    result = ask(zero_shot_cot_prompt, temp=0.0)
    print(f"[Zero-Shot CoT]\n{result}\n")
    # "Think step by step" alone often matches explicit CoT — the model learned this trigger from training.

    # **************************************************************************************************************
    # ── Technique 4: Role Prompting ─────────────────────────────────────────────
    plain_prompt = "Should we use a vector database or a relational database for our RAG system?"

    # Without role — generic answer
    plain_result = ask(plain_prompt)

    # With role — expert framing shifts vocabulary, depth, and tradeoff analysis
    role_result = ask(
        plain_prompt,
        system="You are a senior database architect with 15 years of experience in production AI systems. "
               "Give a direct recommendation with engineering tradeoffs. Be concise."
    )

    print(f"[No Role]\n{plain_result}\n")
    print(f"[With Role]\n{role_result}\n")
    # Observe: the role version typically gives more structured tradeoffs and a clearer recommendation.

    # ************************************************************************************
    # ── Technique 5: Structured Output Contract ─────────────────────────────────

    # Open-ended prompt with no required output format.
    unstructured_prompt = "Compare Chroma, FAISS, and Pinecone as vector databases."


    # No format → model decides (prose, bullet points, or headers — unpredictable)
    # Get the model's free-form comparison.
    unstructured_result = ask(unstructured_prompt)


    # With an explicit output contract → predictable, parseable
    # Prompt that specifies an exact markdown table format.
    structured_prompt = """
    Compare Chroma, FAISS, and Pinecone as vector databases.


    Return your answer as a markdown table with exactly these columns:
    | Database | Type | Best For | Limitation |


    Do not add any text before or after the table.
    """

    # Get a deterministic, format-constrained response.
    structured_result = ask(structured_prompt, temp=0.0)


    # Print only the first 300 characters of the unstructured answer.
    print(f"[No Format]\n{unstructured_result[:300]}...\n")

    # Print the full structured markdown table.
    print(f"[With Format Contract]\n{structured_result}\n")

    # *****************************************************************************
    # ── Technique 6: Self-Consistency ───────────────────────────────────────────
    reasoning_prompt = """
    A RAG pipeline retrieves 5 chunks for a question. 
    Chunk relevance scores (cosine similarity): [0.91, 0.88, 0.72, 0.65, 0.41].
    Should we use the top-3 or top-5 chunks? Give a one-word answer: "top-3" or "top-5".
    """

    votes: list[str] = []
    for _ in range(5):   # 5 independent samples
        answer = ask(reasoning_prompt, temp=0.7)
        # Normalise: extract just "top-3" or "top-5" from the response
        normalised = "top-3" if "top-3" in answer.lower() else "top-5"
        votes.append(normalised)

    majority = max(set(votes), key=votes.count)   # most common answer
    print(f"[Self-Consistency] Votes: {votes}  →  Majority: {majority}")
    # Self-consistency is reliable for this kind of discrete choice question.

    # *********************************************************************************
    # ── Technique 7: System Instruction vs User Message ─────────────────────────
    SYSTEM = (
        "You are a technical documentation assistant for an engineering course. "
        "Rules: (1) Always answer in under 3 sentences. "
        "(2) Use precise technical vocabulary. "
        "(3) If the question is outside your domain, say 'Out of scope.'"
    )

    questions = [
        "What is HNSW and why does Chroma use it?",
        "What is the best recipe for biryani?",     # out-of-scope test
        "What does MMR do in a retrieval pipeline?",
    ]

    for q in questions:
        answer = ask(q, system=SYSTEM)
        print(f"Q: {q}\nA: {answer}\n")

    # The system instruction enforces consistent behavior across all three questions.
    # The out-of-scope question should trigger "Out of scope." — test this live with the class.

    # ***********************************************************************************************8
    # ── Technique 8: Context Window Budgeting ───────────────────────────────────
    import textwrap

    # Simulate retrieved chunks of different lengths
    retrieved_chunks = [
        "Chroma uses HNSW (Hierarchical Navigable Small World) graphs for approximate nearest-neighbour search.",
        "FAISS is a library developed by Meta AI for efficient similarity search and clustering of dense vectors. " * 8,  # long chunk
        "Pinecone is a managed vector database that handles infrastructure, scaling, and replication automatically.",
        "MMR (Maximal Marginal Relevance) balances relevance and diversity when selecting retrieval results.",
        "Cosine similarity measures the angle between two vectors, not their magnitude. Values range from -1 to 1." * 5,  # long chunk
    ]

    MAX_CONTEXT_CHARS = 800    # rough budget for context block in this prompt
    QUESTION = "What is HNSW and how does it relate to Chroma?"

    # Greedy packing: add chunks until budget is exhausted
    packed_chunks = []
    total = 0
    for chunk in retrieved_chunks:
        if total + len(chunk) <= MAX_CONTEXT_CHARS:
            packed_chunks.append(chunk)
            total += len(chunk)
        else:
            # Truncate the chunk to fit remaining budget instead of skipping entirely
            remaining = MAX_CONTEXT_CHARS - total
            if remaining > 100:   # only include if there's meaningful space left
                packed_chunks.append(chunk[:remaining] + "…")
            break

    context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(packed_chunks))

    budgeted_prompt = f"""
    You are a helpful AI assistant. Answer the question using ONLY the context below.
    If the answer is not in the context, say "I don't know."

    Context:
    {context_block}

    Question: {QUESTION}
    """

    answer = ask(budgeted_prompt, temp=0.0)
    print(f"[Budgeted RAG Prompt]\nContext chars used: {total}\nAnswer: {answer}\n")
    # This exact pattern — pack chunks, cite by [n], abstain if not in context — is the RAG prompt