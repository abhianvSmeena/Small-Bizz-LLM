# rag/rag_pipeline.py
import chromadb
from sentence_transformers import SentenceTransformer
import requests

model = SentenceTransformer("BAAI/bge-small-en")
client = chromadb.Client()
collection = client.get_or_create_collection("business_qa")

def retrieve_answer(user_query):
    query_embedding = model.encode(user_query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    context = "\n".join(results['documents'][0])

    prompt = f"Answer this based on context:\n\nContext:\n{context}\n\nQuestion: {user_query}\nAnswer:"

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "phi3", "prompt": prompt, "stream": False}
    )

    return response.json()['response']
