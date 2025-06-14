# embeddings/generate_embeddings.py
import json
import chromadb 
from sentence_transformers import SentenceTransformer

# Load data
with open("data/restaurant_faq.json") as f:
    data = json.load(f)

# Chroma DB
client = chromadb.Client()
collection = client.get_or_create_collection("business_qa")

# Embedding model
model = SentenceTransformer("BAAI/bge-small-en")

for i, item in enumerate(data):
    embedding = model.encode(item["question"])
    collection.add(
        documents=[item["answer"]],
        embeddings=[embedding.tolist()],
        ids=[str(i)]
    )
print("Embeddings created!")
