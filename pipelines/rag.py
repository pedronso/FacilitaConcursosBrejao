import time
import math
import traceback
import pandas as pd
from rank_bm25 import BM25Okapi  # Reranking using BM25
from vectorstore.faiss_store import FAISSVectorStore
from models.llm_model import LLMModel
from models.llm_model import LocalLLMModel
import tests_vars

class RAGPipeline:
    def __init__(self, max_tokens_per_request=2500, max_chunks=tests_vars.dict_models["topk"], tokens_per_minute_limit=6000):
        """
        Initializes the RAG (Retrieval-Augmented Generation) pipeline with reranking.
        
        Args:
        - max_tokens_per_request (int): Maximum tokens allowed per LLM request.
        - max_chunks (int): Maximum number of retrieved chunks from FAISS.
        - tokens_per_minute_limit (int): Rate limit for API token usage.
        """
        self.local_model = LocalLLMModel()
        self.vector_store = FAISSVectorStore()
        self.vector_store.load_indices()
        self.llm = LLMModel()

        # Load processed chunks
        csv_path = "data/processed/results_extraction_chunks_updated.csv"
        self.df_original = pd.read_csv(csv_path)

        if "Concurso" not in self.df_original.columns:
            raise ValueError("❌ Error: 'Concurso' column not found in CSV!")

        self.df_chunks = self.df_original[['Chunk', 'Concurso']].dropna().reset_index(drop=True)

        self.max_tokens_per_request = max_tokens_per_request
        self.max_chunks = max_chunks
        self.tokens_per_minute_limit = tokens_per_minute_limit
        self.tokens_used = 0
        self.start_time = time.time()

    def reset_token_usage_if_needed(self):
        """Resets the token usage counter every minute."""
        if time.time() - self.start_time >= 60:
            print("⏳ Resetting token counter...")
            self.tokens_used = 0
            self.start_time = time.time()

    def wait_if_needed(self):
        """Pauses execution if the API token usage exceeds the limit per minute."""
        while self.tokens_used >= self.tokens_per_minute_limit:
            print(f"🚨 Token limit reached ({self.tokens_used}/{self.tokens_per_minute_limit}). Waiting 30s...")
            time.sleep(30)
            self.reset_token_usage_if_needed()

    def detect_contest(self, query):
        """
        Automatically detects if a contest is mentioned in the query.

        Args:
        - query (str): The user's question.

        Returns:
        - str: Contest name if detected, otherwise None.
        """
        available_contests = self.df_chunks["Concurso"].unique()
        for contest in available_contests:
            if contest.lower() in query.lower():
                return contest
        return None

    def rerank_chunks(self, query, retrieved_chunks):
        """
        Uses BM25 to rerank the retrieved FAISS chunks.

        Args:
        - query (str): The user's question.
        - retrieved_chunks (list of str): Retrieved chunks from FAISS.

        Returns:
        - list of str: Reranked chunks.
        """
        if not retrieved_chunks:
            return []

        tokenized_chunks = [chunk.split() for chunk in retrieved_chunks]
        bm25 = BM25Okapi(tokenized_chunks)

        scores = bm25.get_scores(query.split())
        ranked_chunks = [retrieved_chunks[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)]

        print(f"🔹 Top reranked chunks:\n{ranked_chunks[:3]}")  # Show top 3 chunks for debugging
        return ranked_chunks[:self.max_chunks]

    def generate_answer(self, query):
        """
        Retrieves relevant chunks using FAISS, reranks them, and generates an answer via LLM.

        Args:
        - query (str): The user's question.

        Returns:
        - str: The generated response.
        """
        contest = self.detect_contest(query)

        if contest:
            print(f"🔍 Contest detected: {contest}. Searching in the corresponding index...")
            retrieved_chunks = self.vector_store.search(query, contest)
        else:
            print("🔍 No specific contest detected. Searching globally...")
            retrieved_chunks = self.vector_store.search(query, contest=None)

        # If no relevant chunks are found, return a fallback message
        if not retrieved_chunks or retrieved_chunks == ["❌ No relevant information found for this contest."]:
            return "❌ No relevant information found in the available contest documents."

        # Rerank retrieved chunks
        ranked_chunks = self.rerank_chunks(query, retrieved_chunks)

        # Prepare retrieved text snippets for LLM processing
        relevant_texts = "\n\n".join(ranked_chunks)
        print(f"🔹 Extracted & Reranked Snippets:\n{relevant_texts[:500]}...")

        # Construct prompt for LLM
        prompt = f"""
        Based ONLY on the following excerpts from official contest documents {f'for {contest}' if contest else 'in general'}:

        {relevant_texts}

        Answer the following question as clearly and concisely as possible: {query}

        If the provided excerpts do not contain the exact answer, indicate that the information was not found in the document
        and suggest checking the official contest notice for further details.
        """

        try:
            response = self.llm.generate_response(prompt)
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            traceback.print_exc()
            response = "⚠️ Unable to generate a valid response due to a connection error."

        return response if response else "⚠️ No response generated. Please try again later."

    def generate_full_answer(self, query):
        """
        Retrieves the most relevant chunks, reranks them, and generates a detailed response.

        Args:
        - query (str): The user's question.

        Returns:
        - str: The generated response.
        """
        contest = self.detect_contest(query)
        if contest:
            print(f"🔍 Searching for contest: {contest}...")
            retrieved_chunks = self.vector_store.search(query, contest)
        else:
            print("🔍 No contest detected. Searching globally...")
            retrieved_chunks = self.vector_store.search(query, contest=None)

        if not retrieved_chunks:
            return "❌ No relevant information found."

        # Rerank retrieved chunks
        ranked_chunks = self.rerank_chunks(query, retrieved_chunks)

        relevant_texts = " ".join(ranked_chunks)
        print(f"🔹 Extracted & Reranked Chunks:\n{relevant_texts[:500]}")

        prompt = f"""
        Based ONLY on the following excerpts from official documents:

        {relevant_texts}

        Provide a clear and precise answer to the question: {query}
        """

        response = self.local_model.generate_response(prompt)
        return response

# Test script
if __name__ == "__main__":
    rag = RAGPipeline(max_tokens_per_request=2500, max_chunks=15, tokens_per_minute_limit=6000)
    test_query = "How many vacancies are available in the IBAMA contest?"
    print(rag.generate_full_answer(test_query))
