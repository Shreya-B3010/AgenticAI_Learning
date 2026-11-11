
import os
from typing import List
from dotenv import load_dotenv
from google import genai
import ollama

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_EMBED_MODEL = "gemini-embedding-exp-03-07"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

EMBED_DIM = 768


def embed_gemini(texts: List[str]) -> List[List[float]]:
    """Generate document embeddings using Gemini."""

    embeddings = []

    for text in texts:
        resp = client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=text,
            config={"task_type": "RETRIEVAL_DOCUMENT"}
        )

        embeddings.append(resp.embeddings[0].values)

    return embeddings


def embed_query_gemini(query: str) -> List[float]:
    """Generate a query embedding using Gemini."""

    resp = client.models.embed_content(
        model=GEMINI_EMBED_MODEL,
        contents=query,
        config={"task_type": "RETRIEVAL_QUERY"}
    )

    return resp.embeddings[0].values


def embed_ollama(texts: List[str]) -> List[List[float]]:
    """Generate document embeddings using Ollama."""

    embeddings = []

    for text in texts:
        resp = ollama.embed(
            model=OLLAMA_EMBED_MODEL,
            input=text
        )

        embeddings.append(resp["embeddings"][0])

    return embeddings


def embed_query_ollama(query: str) -> List[float]:
    """Generate a query embedding using Ollama."""

    resp = ollama.embed(
        model=OLLAMA_EMBED_MODEL,
        input=query
    )

    return resp["embeddings"][0]