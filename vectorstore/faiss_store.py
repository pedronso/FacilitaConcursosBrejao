import faiss
import numpy as np
import os
import pandas as pd
from models.embeddings_model import EmbeddingModel

class FAISSVectorStore:
    def __init__(self, index_path):
        """
        Inicializa a classe para um índice FAISS específico.

        Args:
            index_path (str): Caminho do índice FAISS.
        """
    def __init__(self, index_path="data/embeddings/faiss_index"):
        self.index_path = index_path
        self.embedding_model = EmbeddingModel()
        self.index = None

        # Se o índice FAISS não existir, cria um novo
        if not os.path.exists(self.index_path):
            print("⚠️ FAISS index não encontrado! Criando novo...")
            self.create_index_from_chunks()

    def create_index(self, texts):
        """Cria um índice FAISS a partir de uma lista de textos."""
        print(f"🔹 Criando FAISS index em: {self.index_path}")
        #print(len(texts))
        embeddings = [self.embedding_model.get_embedding(" ".join(text) if isinstance(text, list) else text) for text in texts]
        embeddings = np.array(embeddings, dtype=np.float32)

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        print("✅ FAISS index criado e salvo.")

    def create_index_from_chunks(self):
        """Cria um índice FAISS a partir do CSV `results_extraction_chunks.csv`."""
        csv_path = "data/processed/results_extraction_chunks.csv"
        if os.path.exists(csv_path):
            df_chunks = pd.read_csv(csv_path)
            self.create_index(df_chunks["Chunk"].tolist())
        else:
            print("❌ Erro: Nenhum arquivo de chunks encontrado. Rode `extractor.py` primeiro.")

    def load_index(self):
        """Carrega o índice FAISS salvo."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print("✅ FAISS index carregado.")
        else:
            print("❌ Erro: FAISS index não encontrado! Execute `create_index_from_chunks()` primeiro.")

    def search(self, query, k=5):
        """Faz busca no FAISS index."""
        query_embedding = np.array([self.embedding_model.get_embedding(query)], dtype=np.float32)
        _, indices = self.index.search(query_embedding, k)
        return indices[0]

# Teste
if __name__ == "__main__":
    store = FAISSVectorStore()
    store.load_index()  # Se o índice não existir, ele criará um novo automaticamente
    query = "Quais concursos estão com inscrições abertas?"
    results = store.search(query)
    print(f"Índices encontrados: {results}")
