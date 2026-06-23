import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GOOGLE_API_KEY")

print("KEY FOUND:", key is not None)
print("FIRST 10 CHARS:", key[:10] if key else "NONE")
print("LENGTH:", len(key) if key else 0)