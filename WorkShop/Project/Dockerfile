# Day 6 : Block A — containerize the SAME main.py (zero code changes).
# python-slim: the common denominator; no OS tools the agent doesn't need.
FROM python:3.12-slim

WORKDIR /app

# Layer 1 — dependencies first: this layer is CACHED and only re-runs when
# requirements.txt changes (code edits rebuild in seconds, not minutes).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layer 2 — code + corpus. chroma_db MUST be inside the image: the Day 3
# engine opens ./chroma_db relative to the working directory.
COPY . .

# The service listens on 8000 inside the container (-p maps it to your host).
EXPOSE 8000

# python -m uvicorn (not bare `uvicorn`): -m guarantees the working directory
# is on sys.path, so `import main` and the Day2/Day3/Day4 copies resolve.
# ${PORT:-8000}: Vercel's container runtime routes to $PORT (set PORT=8000 in
# the project env); locally PORT is unset -> 8000, exactly as before.
CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]