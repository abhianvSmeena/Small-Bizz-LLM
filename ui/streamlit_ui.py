# ui/streamlit_ui.py
import streamlit as st
import requests

st.title("Small Business AI Chatbot 🤖")

user_input = st.text_input("Ask your question:")

if st.button("Get Answer"):
    response = requests.post("http://localhost:8000/chat", json={"user_query": user_input})
    st.write("Bot:", response.json()["answer"])
