import traceback
import faiss
import numpy as np
import os
import pandas as pd
from models.embeddings_model import EmbeddingModel
from sentence_transformers import SentenceTransformer, util

class FAISSVectorStore:
    def __init__(self, index_path="data/embeddings/faiss_index", reranker_model="sentence-transformers/msmarco-distilbert-base-v4"):
        self.index_path = index_path
        self.embedding_model = EmbeddingModel()
        self.reranker = SentenceTransformer(reranker_model)
        self.index = None

        # Se o índice FAISS não existir, cria um novo
        if not os.path.exists(self.index_path):
            print("⚠️ FAISS index não encontrado! Criando novo...")
            self.create_index_from_chunks()

    def create_index(self, texts):
        """Cria um índice FAISS a partir de uma lista de textos."""
        print(f"🔹 Criando embeddings para {len(texts)} chunks...")
        embeddings = [self.embedding_model.get_embedding(text) for text in texts]
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
            self.create_index(df_chunks["Chunk"].dropna().tolist())
        else:
            print("❌ Erro: Nenhum arquivo de chunks encontrado. Rode `extractor.py` primeiro.")

    def load_index(self):
        """Carrega o índice FAISS salvo."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print("✅ FAISS index carregado.")
        else:
            print("❌ Erro: FAISS index não encontrado! Execute `create_index_from_chunks()` primeiro.")

    def search(self, query, k=15, rerank_top_n=5):
        """Faz busca no FAISS index e aplica re-ranking nos melhores resultados."""
        try:
            query_embedding = np.array([self.embedding_model.get_embedding(query)], dtype=np.float32)
            distances, indices = self.index.search(query_embedding, k)
            
            print(f"🔍 FAISS Retornou Índices: {indices}")
            print(f"🔍 FAISS Retornou Distâncias: {distances}")
        
            if len(indices[0]) == 0:
                return []

            # Recupera os chunks correspondentes aos índices encontrados
            df_chunks = pd.read_csv("data/processed/results_extraction_chunks.csv")
            retrieved_texts = [df_chunks.iloc[i]["Chunk"] for i in indices[0] if i < len(df_chunks)]

            # Re-ranking com Sentence Transformers
            rerank_scores = util.cos_sim(self.reranker.encode(query, convert_to_tensor=True),
                                        self.reranker.encode(retrieved_texts, convert_to_tensor=True))
            
            ranked_results = sorted(zip(retrieved_texts, rerank_scores.tolist()), key=lambda x: x[1], reverse=True)

            # Retorna os top N chunks após re-ranking
            best_chunks = [text for text, _ in ranked_results[:rerank_top_n]]
            return best_chunks
        except Exception as e:
            print(f"❌ Erro ao buscar no FAISS: {e}")
            traceback.print_exc()
            return []
    
    def print_faiss_status(self):
        if self.index is None:
            print("❌ FAISS index não foi inicializado.")
        else:
            print(f"✅ FAISS index contém {self.index.ntotal} embeddings.")


# Teste
if __name__ == "__main__":
    store = FAISSVectorStore()
    store.load_index()
    query = "Quais concursos estão com inscrições abertas?"
    results = store.search(query)
    print(f"🔍 Melhores Chunks: {results}")
