import streamlit as st
import torch
from pipelines.rag import RAGPipeline

torch.classes.__path__ = []

st.title("🤖 Chatbot de Editais de Concursos")

# Inicializa a sessão de histórico se ainda não existir
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir o histórico de mensagens
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

estado = st.toggle("Modo Local", value=False)

query = st.chat_input("Digite sua pergunta...")
rag = RAGPipeline()

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Consultando..."):
            resposta = rag.generate_answer(query) if not estado else rag.generate_full_answer(query)
        
        st.markdown(resposta)

    st.session_state.messages.append({"role": "assistant", "content": resposta})
