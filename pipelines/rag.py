import time
import math
from vectorstore.faiss_store import FAISSVectorStore
from models.llm_model import LLMModel
import pandas as pd

class RAGPipeline:
    def __init__(self, max_tokens_per_request=2500, max_chunks=5, tokens_per_minute_limit=6000):
        """
        Inicializa a pipeline RAG.
        - max_tokens_per_request: número máximo de tokens enviados para o LLM por requisição.
        - max_chunks: número máximo de chunks recuperados do FAISS.
        - tokens_per_minute_limit: limite de tokens por minuto imposto pela API.
        """
        self.vector_store = FAISSVectorStore()
        self.vector_store.load_index()
        self.llm = LLMModel()
        self.df_chunks = pd.read_csv("data/processed/results_extraction_chunks.csv")
        self.max_tokens_per_request = max_tokens_per_request
        self.max_chunks = max_chunks
        self.tokens_per_minute_limit = tokens_per_minute_limit
        self.tokens_used = 0  # Contador de tokens usados
        self.start_time = time.time()  # Início da contagem de tempo

    def reset_token_usage_if_needed(self):
        """Reseta o contador de tokens após 1 minuto."""
        if time.time() - self.start_time >= 60:
            print("⏳ Resetando o contador de tokens...")
            self.tokens_used = 0
            self.start_time = time.time()

    def wait_if_needed(self):
        """Aguarda se a quantidade de tokens usados excedeu o limite por minuto."""
        while self.tokens_used >= self.tokens_per_minute_limit:
            print(f"🚨 Limite de tokens atingido ({self.tokens_used}/{self.tokens_per_minute_limit}). Aguardando 30s...")
            time.sleep(30)  # Aguarda 30s antes de tentar novamente
            self.reset_token_usage_if_needed()

    def split_text(self, text, max_length):
        """Divide um texto em partes que não excedam `max_length` tokens."""
        words = text.split()
        num_parts = math.ceil(len(words) / max_length)
        return [" ".join(words[i * max_length:(i + 1) * max_length]) for i in range(num_parts)]

    def generate_answer(self, query):
        """Busca os chunks relevantes e gera resposta via LLM em partes para evitar perda de dados."""
        indices = self.vector_store.search(query, k=self.max_chunks)
        textos_relevantes = " ".join([self.df_chunks.iloc[i]["Chunks"] for i in indices])

        # Divide o texto excedente em partes de no máximo `max_tokens_per_request`
        partes_texto = self.split_text(textos_relevantes, self.max_tokens_per_request)

        respostas = []
        for i, parte in enumerate(partes_texto):
            self.wait_if_needed()  # Aguarda se necessário antes de enviar mais tokens

            print(f"🔹 Enviando parte {i+1}/{len(partes_texto)} para o LLM...")
            prompt = f"Baseando-se nos seguintes textos, responda:\n{parte}\n\nPergunta: {query}"
            resposta = self.llm.generate_response(prompt)
            respostas.append(resposta)

            # Atualiza o contador de tokens usados
            self.tokens_used += self.max_tokens_per_request

            self.reset_token_usage_if_needed()  # Reseta tokens se o tempo já passou

        return " ".join(respostas)  # Junta todas as partes da resposta final

# Teste Unitário
if __name__ == "__main__":
    rag = RAGPipeline(max_tokens_per_request=2500, max_chunks=5, tokens_per_minute_limit=6000)
    query = "Quais concursos estão com inscrições abertas?"
    print(rag.generate_answer(query))
