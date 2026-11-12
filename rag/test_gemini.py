import os
from dotenv import load_dotenv
from google import genai

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

print("Testing Gemini generation...")

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Explain the 4-3-3 football formation in two sentences."
)

print("\nGemini Response:")
print(response.text)