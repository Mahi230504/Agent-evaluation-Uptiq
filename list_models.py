import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
print(f"Using key starting with: {key[:10]}...")

try:
    client = genai.Client(api_key=key)
    print("Listing all models:")
    # The SDK changed - let's see what's actually in 'client.models.list()'
    for m in client.models.list():
        print(f" - Name: {m.name}")
except Exception as e:
    print(f"Error: {e}")
