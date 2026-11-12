
import os
import chromadb

from dotenv import load_dotenv
from google import genai
from google.genai import types

# --------------------------------
# 1. PROJECT PATHS
# --------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Gemini API key not found!")

# --------------------------------
# 2. CONNECT TO GEMINI
# --------------------------------

client = genai.Client(api_key=API_KEY)

# --------------------------------
# 3. CONNECT TO CHROMADB
# --------------------------------

DB_PATH = os.path.join(BASE_DIR, "rag", "vector_store")

chroma_client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = chroma_client.get_collection(
    name="footiq_knowledge"
)

# --------------------------------
# 4. USER QUESTION
# --------------------------------

question = input("\nAsk FootIQ a football question: ")

# --------------------------------
# 5. GENERATE QUESTION EMBEDDING
# --------------------------------

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question,
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY"
    )
)

query_embedding = response.embeddings[0].values

# --------------------------------
# 6. SEARCH CHROMADB
# --------------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

# --------------------------------
# 7. DISPLAY RETRIEVED KNOWLEDGE
# --------------------------------

print("\n========== FOOTIQ SEARCH RESULTS ==========")

for i, (document, metadata, distance) in enumerate(
    zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ),
    start=1
):
    print(f"\nResult {i}")
    print("Topic:", metadata["topic"])
    print("Distance:", round(distance, 4))
    print("Information:", document.strip())

print("\n===========================================")
