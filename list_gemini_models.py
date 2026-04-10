import os
from google import genai
from dotenv import load_dotenv
from pathlib import Path

def list_models():
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env", override=True)
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("No GEMINI_API_KEY found in .env")
        return
    
    client = genai.Client(api_key=key)
    print(f"Using key: {key[:5]}...{key[-5:]}")
    print("Available models:")
    try:
        for m in client.models.list():
            print(f"- {m.name} (Supported methods: {m.supported_generation_methods})")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_models()
