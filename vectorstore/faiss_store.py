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

        # Carregar os chunks processados
        csv_path = "data/processed/results_extraction_chunks_updated.csv"
        if os.path.exists(csv_path):
            self.df_chunks = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"❌ Erro: Arquivo CSV {csv_path} não encontrado!")

        # Criar um DataFrame com os intervalos de índices por concurso
        self.concurso_indices = self._calcular_indices_concursos()

        # Se o índice FAISS não existir, cria um novo
        if not os.path.exists(self.index_path):
            print("⚠️ FAISS index não encontrado! Criando novo...")
            self.create_index_from_chunks()

    def _calcular_indices_concursos(self):
        """Cria um dicionário com os intervalos de índices de cada concurso."""
        concursos = self.df_chunks["Concurso"].dropna().unique()
        concurso_indices = {}

        print("\nconcursos: ", concursos)
        for concurso in concursos:
            indices = self.df_chunks[self.df_chunks["Concurso"] == concurso].index
            if len(indices) > 0:
                concurso_indices[concurso] = (indices.min(), indices.max())

        return concurso_indices

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
        """Cria um índice FAISS a partir do CSV `results_extraction_chunks_updated.csv`."""
        csv_path = "data/processed/results_extraction_chunks_updated.csv"
        if os.path.exists(csv_path):
            self.create_index(self.df_chunks["Chunk"].dropna().tolist())
        else:
            print("❌ Erro: Nenhum arquivo de chunks encontrado. Rode `extractor.py` primeiro.")

    def load_index(self):
        """Carrega o índice FAISS salvo."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print("✅ FAISS index carregado.")
        else:
            print("❌ Erro: FAISS index não encontrado! Execute `create_index_from_chunks()` primeiro.")

    def search(self, query, concurso=None, k=50, rerank_top_n=10):
        """Faz busca no FAISS, priorizando o concurso específico e ampliando para global se necessário."""
        try:
            query_embedding = np.array([self.embedding_model.get_embedding(query)], dtype=np.float32)
            distances, indices = self.index.search(query_embedding, k)

            print(f"🔍 FAISS Retornou Índices: {indices}")
            print(f"🔍 FAISS Retornou Distâncias: {distances}")

            filtered_indices = indices[0].tolist()
            retrieved_texts = []

            # 🔹 Filtro de concurso, se houver um identificado
            if concurso and concurso in self.concurso_indices:
                min_index, max_index = self.concurso_indices[concurso]
                filtered_indices = [i for i in filtered_indices if min_index <= i <= max_index]

            retrieved_texts = [self.df_chunks.iloc[i]["Chunk"] for i in filtered_indices if i < len(self.df_chunks)]

            # 🔹 Se não encontrou nada, buscar globalmente
            if not retrieved_texts:
                print(f"⚠️ Nenhum chunk encontrado para {concurso}. Tentando busca global...")
                filtered_indices = indices[0].tolist()
                retrieved_texts = [self.df_chunks.iloc[i]["Chunk"] for i in filtered_indices if i < len(self.df_chunks)]

            if not retrieved_texts:
                return ["❌ Nenhuma informação específica encontrada para esse concurso."]

            # 🔹 Melhorando o reranking com modelo mais avançado
            rerank_scores = util.cos_sim(
                self.reranker.encode(query, convert_to_tensor=True),
                self.reranker.encode(retrieved_texts, convert_to_tensor=True)
            )
            ranked_results = sorted(zip(retrieved_texts, rerank_scores.tolist()), key=lambda x: x[1], reverse=True)

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
    query = "Quantas vagas estão abertas?"
    concurso = "FUNAI"
    results = store.search(query, concurso)
    print(f"🔍 Melhores Chunks para {concurso}: {results}")
