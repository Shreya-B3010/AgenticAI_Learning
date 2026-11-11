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
    model="gemini-2.5-flash",
    contents="Reply with exactly three words:API key works."
)
print(f"[PING]{ping.text.strip()}")

#Test the ollama based LLM
import ollama
GEMINI_MODEL = "gemini-2.5-flash"
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

# ******************************************
# Prompt used to test token-by-token streaming from both providers.
STREAM_PROMPT = "List 5 real-world use cases of Retrieval-Augmented Generation, one per line."


# Print a section header for Gemini streaming.
print("\n── Gemini Streaming ──")

# stream=True returns an iterator of response chunks instead of one complete object.
# Iterate over Gemini's streamed response chunks.
for chunk in client.models.generate_content_stream(
    model=GEMINI_MODEL,
    contents=STREAM_PROMPT,
):
    # chunk.text is the new text fragment in this chunk; may be None on the final metadata chunk.
    # Print only chunks that contain text.
    if chunk.text:
        print(chunk.text, end="", flush=True)
        # end="" prevents a newline between chunks.
        # flush=True forces the terminal to display the text immediately (not buffer it).

# Add a newline once streaming completes.
print()  # newline after stream ends


# Print a section header for Ollama streaming.
print("\n── Ollama Streaming ──")

# stream=True in ollama.chat also returns a generator.
# Iterate over Ollama's streamed response chunks.
for chunk in ollama.chat(
    model=OLLAMA_MODEL,
    messages=[{"role": "user", "content": STREAM_PROMPT}],
    stream=True,
):
    # Extract the new text token/fragment from this chunk.
    token = chunk["message"]["content"]

    # Print it immediately without adding a newline.
    print(token, end="", flush=True)

# Add a newline once Ollama streaming completes.
print()

# **************************************************************
# Prompt asking Gemini to summarize a paragraph and return structured JSON.
JSON_PROMPT = """
Summarize the following paragraph.
Return a JSON object with exactly these keys:
  - "summary": one sentence summary (string)
  - "topics": list of up to 3 key topics (array of strings)
  - "sentiment": one of "positive", "neutral", or "negative" (string)


Paragraph:
\"Retrieval-Augmented Generation (RAG) is a powerful technique that combines
the strengths of dense retrieval with large language models. By grounding LLM
responses in retrieved documents, RAG dramatically reduces hallucinations and
keeps answers factual. It has become the dominant architecture for enterprise
knowledge-base chatbots and Q&A systems in 2024-2026.\"
"""


# Send the prompt to Gemini with JSON output enabled.
json_resp = client.models.generate_content(
    model=GEMINI_MODEL,
    contents=JSON_PROMPT,
    config=types.GenerateContentConfig(
        # constrains the model to emit valid JSON only
        response_mime_type="application/json",

        # low temperature for structured extraction — consistency matters
        temperature=0.1,
    ),
)


# response_mime_type="application/json" makes it safe to call json.loads() directly.
# Parse the model's JSON text into a Python dictionary.
structured = json.loads(json_resp.text)


# Print a section header.
print("\n── JSON Mode Output ──")

# Print the extracted summary.
print(f"Summary  : {structured['summary']}")

# Print the extracted topic list.
print(f"Topics   : {structured['topics']}")

# Print the extracted sentiment.
print(f"Sentiment: {structured['sentiment']}")

# ***********************************************************************
# Benchmark prompt used for both providers.
BENCH_PROMPT = "What is the difference between fine-tuning and RAG? Answer in 2 sentences."

# Number of timing runs.
N_RUNS = 5


# Print benchmark header.
print(f"\n── {N_RUNS}-run Latency Benchmark ──")


# Lists to store each provider's response times.
gemini_times, ollama_times = [], []


# Run the benchmark N_RUNS times.
for i in range(1, N_RUNS + 1):
    # Gemini run
    # Start high-resolution timer.
    t0 = time.perf_counter()

    # Send prompt to Gemini.
    r = client.models.generate_content(model=GEMINI_MODEL, contents=BENCH_PROMPT)

    # Record elapsed time for Gemini.
    gemini_times.append(time.perf_counter() - t0)


    # Ollama run
    # Start timer.
    t0 = time.perf_counter()

    # Send the same prompt to the local Ollama model.
    ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": BENCH_PROMPT}])

    # Record elapsed time for Ollama.
    ollama_times.append(time.perf_counter() - t0)


    # Show timing for the current run.
    print(f"  Run {i}: Gemini={gemini_times[-1]:.2f}s  Ollama={ollama_times[-1]:.2f}s")


# Arithmetic mean of the 5 runs — smooths out outliers from network jitter or cold starts.
# Calculate average latency for Gemini.
gemini_avg = sum(gemini_times) / N_RUNS

# Calculate average latency for Ollama.
ollama_avg = sum(ollama_times) / N_RUNS


# Print average latencies.
print(f"\n  Average — Gemini: {gemini_avg:.2f}s | Ollama: {ollama_avg:.2f}s")

# Print token usage from the final Gemini response.
print(f"  Last Gemini token count: {r.usage_metadata.total_token_count} tokens")