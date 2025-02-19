import traceback
import faiss
import numpy as np
import os
import pandas as pd
from models.embeddings_model import EmbeddingModel
from sentence_transformers import SentenceTransformer, util

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
        self.concurso_indices = {}  # ✅ Now initialized properly
        self.df_chunks = None

        # Load processed chunks
        csv_path = "data/processed/results_extraction_chunks_updated.csv"
        if os.path.exists(csv_path):
            self.df_chunks = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"❌ Error: CSV file {csv_path} not found!")

        # ✅ Now we check if indices exist before deciding to create or load them
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
        """Loads all FAISS indices from disk."""
        if not self.concurso_indices:
            self.concurso_indices = {}  # ✅ Ensures self.concurso_indices is not None

        for contest in self.df_chunks["Concurso"].dropna().unique():
            index_path = os.path.join(self.base_path, f"faiss_index_{contest.lower()}")
            if os.path.exists(index_path):
                self.indices[contest] = faiss.read_index(index_path)
                self.concurso_indices[contest] = index_path  # ✅ Keep track of loaded indices
                print(f"✅ Loaded FAISS index for {contest}.")
            else:
                print(f"⚠️ FAISS index missing for {contest}. Recreating...")
                self.create_indices()

        # Load global index
        global_index_path = os.path.join(self.base_path, "faiss_index_global")
        if os.path.exists(global_index_path):
            self.indices["global"] = faiss.read_index(global_index_path)
            self.concurso_indices["global"] = global_index_path  # ✅ Ensure global index is tracked
            print("✅ Loaded FAISS global index.")
        else:
            print("⚠️ Global FAISS index missing. Recreating...")
            self.create_indices()

    def search(self, query, contest=None, k=50, rerank_top_n=10):
        """
        Searches for relevant chunks using FAISS, prioritizing a contest-specific index if provided.
        If no relevant chunks are found, falls back to the global index.

        Args:
        - query (str): User query.
        - contest (str, optional): Contest to filter search results.
        - k (int): Number of results to retrieve from FAISS.
        - rerank_top_n (int): Number of top results to return after reranking.

        Returns:
        - List of relevant chunks.
        """
        try:
            query_embedding = np.array([self.embedding_model.get_embedding(query)], dtype=np.float32)

            # Step 1: Search contest-specific index (if available)
            if contest and contest in self.indices:
                print(f"🔍 Searching in FAISS index for {contest}...")
                index = self.indices[contest]
            else:
                print(f"⚠️ No index found for {contest}. Searching globally...")
                index = self.indices["global"]

            # Perform FAISS search
            distances, indices = index.search(query_embedding, k)
            print(f"🔍 FAISS returned indices: {indices}")
            print(f"🔍 FAISS returned distances: {distances}")

            # Filter indices to remove invalid ones
            retrieved_indices = [i for i in indices[0] if 0 <= i < len(self.df_chunks)]
            retrieved_texts = [self.df_chunks.iloc[i]["Chunk"] for i in retrieved_indices]

            # Step 2: If no results in contest-specific index, fallback to global search
            if not retrieved_texts and contest:
                print(f"⚠️ No relevant chunks found for {contest}. Falling back to global search...")
                distances, indices = self.indices["global"].search(query_embedding, k)
                retrieved_indices = [i for i in indices[0] if 0 <= i < len(self.df_chunks)]
                retrieved_texts = [self.df_chunks.iloc[i]["Chunk"] for i in retrieved_indices]

            # Step 3: If still no results, return error message
            if not retrieved_texts:
                return ["❌ No relevant information found for this contest."]

            # Step 4: Apply reranking for better precision
            rerank_scores = util.cos_sim(
                self.reranker.encode(query, convert_to_tensor=True),
                self.reranker.encode(retrieved_texts, convert_to_tensor=True)
            )
            ranked_results = sorted(zip(retrieved_texts, rerank_scores.tolist()), key=lambda x: x[1], reverse=True)

            # Return top reranked results
            best_chunks = [text for text, _ in ranked_results[:rerank_top_n]]
            return best_chunks

        except Exception as e:
            print(f"❌ Error in FAISS search: {e}")
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
