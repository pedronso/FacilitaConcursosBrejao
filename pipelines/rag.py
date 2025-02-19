import time
import traceback
import pandas as pd
from rank_bm25 import BM25Okapi  # Reranking using BM25
from vectorstore.faiss_store import FAISSVectorStore
from models.llm_model import LLMModel
from models.llm_model import LocalLLMModel
from utils.contest_detector import detect_contest
import tests_vars

class RAGPipeline:
    def __init__(self, max_tokens_per_request=2500, max_chunks=tests_vars.dict_models["topk"], tokens_per_minute_limit=6000):
        """
        Inicializa o pipeline RAG (Retrieval-Augmented Generation) com reranking.

        Args:
        - max_tokens_per_request (int): Máximo de tokens permitidos por requisição ao LLM.
        - max_chunks (int): Número máximo de trechos recuperados do FAISS.
        - tokens_per_minute_limit (int): Limite de tokens por minuto na API.
        """
        self.local_model = LocalLLMModel()
        self.vector_store = FAISSVectorStore()
        self.vector_store.load_indices()
        self.llm = LLMModel()

        # Carrega os trechos processados
        csv_path = "data/processed/results_extraction_chunks_updated.csv"
        self.df_original = pd.read_csv(csv_path)

        if "Concurso" not in self.df_original.columns:
            raise ValueError("❌ Erro: coluna 'Concurso' não encontrada no CSV!")

        self.df_chunks = self.df_original[['Chunk', 'Concurso']].dropna().reset_index(drop=True)

        self.max_tokens_per_request = max_tokens_per_request
        self.max_chunks = max_chunks
        self.tokens_per_minute_limit = tokens_per_minute_limit
        self.tokens_used = 0
        self.start_time = time.time()

    def reset_token_usage_if_needed(self):
        """Reseta o contador de tokens a cada minuto."""
        if time.time() - self.start_time >= 60:
            print("⏳ Reiniciando contador de tokens...")
            self.tokens_used = 0
            self.start_time = time.time()

    def wait_if_needed(self):
        """Pausa a execução se o uso de tokens atingir o limite por minuto."""
        while self.tokens_used >= self.tokens_per_minute_limit:
            print(f"🚨 Limite de tokens atingido ({self.tokens_used}/{self.tokens_per_minute_limit}). Aguardando 30s...")
            time.sleep(30)
            self.reset_token_usage_if_needed()

    def rerank_chunks(self, query, retrieved_chunks):
        """
        Utiliza BM25 para reranquear os trechos recuperados do FAISS.

        Args:
        - query (str): Pergunta do usuário.
        - retrieved_chunks (list of str): Trechos recuperados do FAISS.

        Returns:
        - list of str: Trechos reranqueados.
        """
        if not retrieved_chunks:
            return []

        tokenized_chunks = [chunk.split() for chunk in retrieved_chunks]
        bm25 = BM25Okapi(tokenized_chunks)

        scores = bm25.get_scores(query.split())
        ranked_chunks = [retrieved_chunks[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)]

        print(f"🔹 Top trechos reranqueados:\n{ranked_chunks[:3]}")  # Exibe os 3 melhores trechos para debug
        return ranked_chunks[:self.max_chunks]

    def detect_multiple_contests(self, query):
        """Detecta múltiplos concursos mencionados na query."""
        contests_detected = [contest for contest in self.df_chunks["Concurso"].unique() if contest.lower() in query.lower()]
        return contests_detected if contests_detected else None

    def generate_answer(self, query):
        """Gera uma resposta baseada nos trechos do FAISS."""
        contests = self.detect_multiple_contests(query)
        
        if not contests:
            print("🚨 [DEBUG] Nenhum concurso detectado na consulta.")
            return "❌ Não consegui identificar a qual concurso você se refere. Reformule sua pergunta."

        responses = []

        for contest in contests:
            print(f"🔍 [DEBUG] Buscando informações para o concurso: {contest}...")
            retrieved_chunks = self.vector_store.search(query, contest)

            if not retrieved_chunks or "❌" in retrieved_chunks[0]:
                print(f"⚠️ [DEBUG] Nenhuma informação relevante encontrada para {contest}.")
                responses.append(f"❌ Nenhuma informação relevante encontrada para o concurso {contest}.")
                continue

            ranked_chunks = self.rerank_chunks(query, retrieved_chunks)
            relevant_texts = "\n\n".join(ranked_chunks)

            prompt = f"""
            Baseando-se APENAS nos seguintes trechos extraídos do edital do concurso {contest}:

            {relevant_texts}

            Responda à seguinte pergunta de forma clara e objetiva: {query}

            Se os trechos fornecidos não contiverem a resposta exata, informe que a informação não foi encontrada e recomende a leitura do edital oficial.
            """

            response = self.llm.generate_response(prompt)
            responses.append(f"📌 **{contest}:** {response if response else '⚠️ Nenhuma resposta gerada.'}")

        return "\n\n".join(responses)


    def generate_full_answer(self, query):
        """
        Recupera os trechos mais relevantes, reranqueia e gera uma resposta detalhada.

        Args:
        - query (str): Pergunta do usuário.

        Returns:
        - str: Resposta gerada.
        """
        contest = detect_contest(query)

        if contest:
            print(f"🔍 Buscando informações sobre o concurso: {contest}...")
            retrieved_chunks = self.vector_store.search(query, contest)
        else:
            return "❌ Não consegui identificar a qual concurso você se refere. Poderia especificar melhor?"

        if not retrieved_chunks:
            return f"❌ Não encontrei informações sobre o concurso {contest}."

        # Reranqueia os trechos recuperados
        ranked_chunks = self.rerank_chunks(query, retrieved_chunks)

        relevant_texts = " ".join(ranked_chunks)
        print(f"🔹 Trechos extraídos & reranqueados:\n{relevant_texts[:500]}")

        prompt = f"""
        Baseando-se APENAS nos seguintes trechos extraídos de documentos oficiais do concurso {contest}:

        {relevant_texts}

        Forneça uma resposta clara e detalhada para a pergunta: {query}
        """

        response = self.local_model.generate_response(prompt)
        return response

# Teste do script
if __name__ == "__main__":
    rag = RAGPipeline(max_tokens_per_request=2500, max_chunks=15, tokens_per_minute_limit=6000)
    test_query = "Quantas vagas estão disponíveis no concurso do IBAMA?"
    print(rag.generate_full_answer(test_query))
