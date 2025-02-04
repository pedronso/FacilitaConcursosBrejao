from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Nova importação correta

class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Inicializa o modelo de embeddings do Hugging Face."""
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def get_embedding(self, text):
        """Gera embeddings para um determinado texto."""
        return self.embedding_model.embed_query(text)

# Teste
if __name__ == "__main__":
    model = EmbeddingModel()
    texto_teste = "Este é um teste para gerar embeddings."
    embedding = model.get_embedding(texto_teste)
    print(f"Embedding gerado: {embedding}...")
