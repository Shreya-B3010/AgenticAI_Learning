import os 
import time
import json

from dotenv import load_dotenv #.env
from google import genai
from google.genai import types 

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not found")

client = genai.Client(api_key=GEMINI_API_KEY)

ping = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Reply with exactly three words:API key works."
)
print(f"[PING]{ping.text.strip()}")

#Test the ollama based LLM
import ollama
GEMINI_MODEL = "gemini-3.6-flash"
OLLAMA_MODEL = "gemma3:4b" #ollama list

BENCHMARK_PROMPT = "Explain what a Transformer neural network is in exactly 3 sentences"

#Gemini call
t0 = time.perf_counter()
gemini_resp = client.models.generate_content(model=GEMINI_MODEL,
                                        contents=BENCHMARK_PROMPT)
gemini_latency = time.perf_counter() - t0
gemini_text = gemini_resp.text.strip()
gemini_tokens = gemini_resp.usage_metadata.total_token_count

#ollama call
t0 = time.perf_counter()
ollama_resp = ollama.chat(model=OLLAMA_MODEL,
                        messages=[{"role":"user","content":BENCHMARK_PROMPT}],)
ollama_latency = time.perf_counter() - t0
ollama_text = ollama_resp["message"]["content"].strip()
ollama_tokens = ollama_resp.get("eval_count","n/a")

print(f"{'MODEL':<25}{'Latency(s)':>12}{'Tokens':>8}")
print(f"{GEMINI_MODEL:<25} {gemini_latency:>12.2f}{gemini_tokens:>8}")
print(f"{OLLAMA_MODEL:<25} {str(ollama_latency)[:5]:>12}{str(ollama_tokens):>8}")

print(f"\nGemini result: {gemini_text}")
print(f"\nOllama result: {ollama_text}")