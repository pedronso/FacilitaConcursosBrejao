import streamlit as st
from pipelines.rag import RAGPipeline

st.title("📜 Chatbot de Editais de Concursos")

st.write("Digite sua pergunta sobre os editais de concursos públicos disponíveis.")

query = st.text_input("Pergunta:")
rag = RAGPipeline()

if st.button("Consultar"):
    if query:
        resposta = rag.generate_answer(query)
        st.write(f"**Resposta:** {resposta}")
    else:
        st.warning("Por favor, insira uma pergunta válida.")
