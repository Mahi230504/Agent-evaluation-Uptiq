import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
print(f"Using key starting with: {key[:10]}...")

try:
    client = genai.Client(api_key=key)
    print("Searching for 1.5 models:")
    for m in client.models.list():
        if "1.5" in m.name:
            print(f" - {m.name}")
except Exception as e:
    print(f"Error: {e}")
