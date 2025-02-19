import traceback
import faiss
import numpy as np
import os
import pandas as pd
from models.embeddings_model import EmbeddingModel
from sentence_transformers import SentenceTransformer, util
from utils.contest_detector import detect_contest

class FAISSVectorStore:
    def __init__(self, base_path="data/embeddings/"):
        """
        Initializes a FAISS vector store with per-contest indices and a global fallback.
        If indices exist, they are loaded. If not, they are created.
        """
        self.base_path = base_path
        self.embedding_model = EmbeddingModel()
        self.reranker = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v4")
        self.indices = {}
        self.concurso_indices = {}  
        self.df_chunks = None

        # Load processed chunks
        csv_path = "data/processed/results_extraction_chunks_updated.csv"
        if os.path.exists(csv_path):
            self.df_chunks = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"❌ Error: CSV file {csv_path} not found!")

        if self._indices_exist():
            self.load_indices()
        else:
            print("⚠️ No FAISS indices found, creating them now...")
            self.create_indices()

    def _indices_exist(self):
        """Checks if FAISS indices already exist."""
        contests = self.df_chunks["Concurso"].dropna().unique()
        for contest in contests:
            index_path = os.path.join(self.base_path, f"faiss_index_{contest.lower()}")
            if not os.path.exists(index_path):
                return False

        # Check global index
        global_index_path = os.path.join(self.base_path, "faiss_index_global")
        return os.path.exists(global_index_path)

    def create_indices(self):
        """Creates a FAISS index for each contest and a global index."""
        contests = self.df_chunks["Concurso"].dropna().unique()
        self.concurso_indices = {}  # ✅ Ensures the attribute exists before use

        print(f"\n🔍 Creating indices for contests: {contests}")
        os.makedirs(self.base_path, exist_ok=True)

        for contest in contests:
            chunks = self.df_chunks[self.df_chunks["Concurso"] == contest]["Chunk"].dropna().tolist()
            index_path = os.path.join(self.base_path, f"faiss_index_{contest.lower()}")

            if chunks:
                print(f"📌 Creating FAISS index for {contest} with {len(chunks)} chunks...")
                self.indices[contest] = self._create_faiss_index(index_path, chunks)
                self.concurso_indices[contest] = index_path  # ✅ Ensures correct tracking

        # Create global index
        global_chunks = self.df_chunks["Chunk"].dropna().tolist()
        global_index_path = os.path.join(self.base_path, "faiss_index_global")
        self.indices["global"] = self._create_faiss_index(global_index_path, global_chunks)
        self.concurso_indices["global"] = global_index_path  # ✅ Add global index path

    def _create_faiss_index(self, index_path, texts):
        """Creates a FAISS index for the given texts and saves it to disk."""
        print(f"🔹 Generating embeddings for {len(texts)} chunks...")
        embeddings = [self.embedding_model.get_embedding(text) for text in texts]
        embeddings = np.array(embeddings, dtype=np.float32)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, index_path)
        print(f"✅ FAISS index saved at: {index_path}")

        return index

    def load_indices(self):
        """Carrega todos os índices FAISS disponíveis."""
        if not self.concurso_indices:
            self.concurso_indices = {}

        print("🔍 [DEBUG] Carregando índices FAISS...")

        for contest in self.df_chunks["Concurso"].dropna().unique():
            index_path = os.path.join(self.base_path, f"faiss_index_{contest.lower()}")
            if os.path.exists(index_path):
                self.indices[contest] = faiss.read_index(index_path)
                self.concurso_indices[contest] = index_path
                print(f"✅ [DEBUG] Índice FAISS carregado para {contest}. Total de embeddings: {self.indices[contest].ntotal}")
            else:
                print(f"⚠️ [DEBUG] Índice FAISS ausente para {contest}. Recriando...")

        # Carregar índice global
        global_index_path = os.path.join(self.base_path, "faiss_index_global")
        if os.path.exists(global_index_path):
            self.indices["global"] = faiss.read_index(global_index_path)
            self.concurso_indices["global"] = global_index_path
            print(f"✅ [DEBUG] Índice FAISS Global carregado. Total de embeddings: {self.indices['global'].ntotal}")
        else:
            print("⚠️ [DEBUG] Índice FAISS Global ausente. Recriando...")


    def search(self, query, contest=None, k=50, rerank_top_n=10):
        """Busca trechos relevantes no índice FAISS correto."""
        try:
            contest_detected = detect_contest(query) if contest is None else contest
            if not contest_detected:
                print("🚨 [DEBUG] Nenhum concurso detectado na consulta.")
                return ["❌ Não consegui identificar a qual concurso você se refere. Reformule sua pergunta."]

            print(f"🔍 [DEBUG] Concurso detectado: {contest_detected}")

            query_embedding = np.array([self.embedding_model.get_embedding(query)], dtype=np.float32)

            # Verifica se o índice FAISS do concurso existe
            if contest_detected in self.indices:
                index = self.indices[contest_detected]
                print(f"✅ [DEBUG] Índice FAISS carregado para {contest_detected}.")
                print(f"📊 [DEBUG] Número de embeddings no índice: {index.ntotal}")
            else:
                print(f"❌ [DEBUG] Nenhum índice FAISS disponível para {contest_detected}.")
                return [f"❌ Não há um índice FAISS disponível para o concurso {contest_detected}."]

            # Realiza a busca no índice FAISS
            distances, indices = index.search(query_embedding, k)
            print(f"🔍 [DEBUG] Índices retornados pelo FAISS: {indices}")

            # Garante que os trechos recuperados pertencem ao concurso correto
            retrieved_indices = [
                i for i in indices[0] if 0 <= i < len(self.df_chunks) and 
                self.df_chunks.iloc[i]["Concurso"] == contest_detected
            ]
            retrieved_texts = [self.df_chunks.iloc[i]["Chunk"] for i in retrieved_indices]

            print(f"🔹 [DEBUG] Trechos recuperados após filtragem: {len(retrieved_texts)}")

            if not retrieved_texts:
                return [f"❌ Nenhuma informação relevante encontrada para o concurso {contest_detected}."]

            # Reranking dos trechos retornados
            rerank_scores = util.cos_sim(
                self.reranker.encode(query, convert_to_tensor=True),
                self.reranker.encode(retrieved_texts, convert_to_tensor=True)
            )
            ranked_results = sorted(zip(retrieved_texts, rerank_scores.tolist()), key=lambda x: x[1], reverse=True)
            best_chunks = [text for text, _ in ranked_results[:rerank_top_n]]

            return best_chunks

        except Exception as e:
            print(f"❌ [DEBUG] Erro na busca FAISS: {e}")
            traceback.print_exc()
            return []



    def print_faiss_status(self):
        """Prints the status of FAISS indices."""
        for contest, index in self.indices.items():
            print(f"✅ {contest} FAISS index contains {index.ntotal} embeddings.")

# Test script
if __name__ == "__main__":
    store = FAISSVectorStore()
    store.load_indices()
    query = "How many open positions are available?"
    contest = "FUNAI"
    results = store.search(query, contest)
    print(f"🔍 Best chunks for {contest}: {results}")
