import os
import sys
import subprocess
from pathlib import Path

def run_diagnostic():
    print("--- Gemini API Diagnostic ---")
    
    # 1. Load context
    project_root = Path(__file__).resolve().parent
    env_path = project_root / ".env"
    
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)
    
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("❌ ERROR: No GEMINI_API_KEY found in .env")
        return

    print(f"✅ Key found: {key[:5]}...{key[-5:]}")

    # 2. Try simple CURL (Most reliable check)
    print("\n--- Testing via CURL (v1beta) ---")
    curl_cmd = f"curl -s -X GET https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        import json
        result = subprocess.check_output(curl_cmd, shell=True).decode('utf-8')
        data = json.loads(result)
        if 'models' in data:
            print(f"✅ Success! Found {len(data['models'])} models.")
            flash_models = [m['name'] for m in data['models'] if 'flash' in m['name'].lower()]
            print(f"Available Flash Models: {flash_models}")
        else:
            print(f"❌ API Error: {data}")
    except Exception as e:
        print(f"❌ CURL failed: {e}")

    # 3. Try to check SDK
    print("\n--- Testing via SDK ---")
    try:
        from google import genai
        client = genai.Client(api_key=key)
        print("✅ google-genai SDK found.")
        
        test_models = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-flash-001", "gemini-2.0-flash"]
        for m_name in test_models:
            print(f"Testing {m_name}: ", end="", flush=True)
            try:
                # Just check if we can get model metadata
                m = client.models.get(model=m_name)
                print("✅ Found!")
            except Exception as e:
                print(f"❌ Failed: {e}")
    except ImportError:
        print("⚠️ google-genai SDK not installed or 'google' namespace blocked.")
    except Exception as e:
        print(f"❌ SDK Diagnostic failed: {e}")

if __name__ == "__main__":
    run_diagnostic()
