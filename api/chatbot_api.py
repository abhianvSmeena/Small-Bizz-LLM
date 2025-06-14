# api/chatbot_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import retrieve_answer

app = FastAPI()

class Query(BaseModel):
    user_query: str

@app.post("/chat")
async def chat(query: Query):
    answer = retrieve_answer(query.user_query)
    return {"answer": answer}
