import os
import time
import json

from dotenv import load_dotenv #.env
from google import genai
from google.genai import types #access the google genrative AI API

load_dotenv() #load the .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") #get the API key from the .env file

if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY is not found")

client = genai.Client(api_key=GEMINI_API_KEY) #initialize

ping = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Reply with exactly three words:API key Works."
)
print(f"[PING]{ping.text.strip()}")