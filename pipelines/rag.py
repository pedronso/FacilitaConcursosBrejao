import time
import math
from vectorstore.faiss_store import FAISSVectorStore
from models.llm_model import LLMModel
from models.llm_model import LocalLLMModel
import pandas as pd
import tests_vars

class RAGPipeline:
    def __init__(self, max_tokens_per_request=2500, max_chunks=tests_vars.dict_models["topk"], tokens_per_minute_limit=6000):
        """
        Inicializa a pipeline RAG.
        - max_tokens_per_request: número máximo de tokens enviados para o LLM por requisição.
        - max_chunks: número máximo de chunks recuperados do FAISS.
        - tokens_per_minute_limit: limite de tokens por minuto imposto pela API.
        """
        self.local_model = LocalLLMModel()
        self.vector_store = FAISSVectorStore()
        self.vector_store.load_index()
        self.llm = LLMModel()
        self.df_original = pd.read_csv("data/processed/results_extraction_chunks.csv")
        self.df_chunks = self.df_original.dropna().reset_index(drop=True)
        #self.df_chunks = self.df_original.melt(var_name="Chunk_Index", value_name="Chunk").dropna().reset_index(drop=True)

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
        #print(self.max_chunks)
        indices = [int(i) for i in indices if 0 <= i < len(self.df_chunks)]
        textos_relevantes = " ".join([self.df_chunks.iloc[i]["Chunk"] for i in indices])

        # print(textos_relevantes)
        print(indices)
        # Divide o texto excedente em partes de no máximo `max_tokens_per_request`
        #partes_texto = self.split_text(textos_relevantes, self.max_tokens_per_request)
        
        prompt = f"Baseando-se somente nos seguintes textos:{textos_relevantes}\n\n responda: {query}"
        resposta = self.llm.generate_response(prompt)

        return resposta
    
    def generate_full_answer(self, query):
        """
        print("pegando indices...")
        indices = self.vector_store.search(query, k=self.max_chunks)
        indices = [int(i) for i in indices if 0 <= i < len(self.df_chunks)]
        print(f'pegado {indices}')

        textos_relevantes = " ".join([self.df_chunks.iloc[i]["Chunks"] for i in indices])
        """
        indices = self.vector_store.search(query, k=self.max_chunks)
        indices = [int(i) for i in indices if 0 <= i < len(self.df_chunks)]
        print(f'Pegando índices: {indices}')

        textos_relevantes = " ".join([self.df_chunks.iloc[i]["Chunk"] for i in indices])
        """
        textos_relevantes = " ".join([
            " ".join(map(str, self.df_chunks.iloc[i].dropna())) for i in indices
        ])
        """

        #print(f"Gerando resposta com texto inteiro...{textos_relevantes}")

        prompt = f"Baseando-se somente nos seguintes textos:\n{textos_relevantes}\n\n responda: {query}"
        #prompt = f"Me diga o que tem nos seguintes textos separando cada um:\n{textos_relevantes}"
        prompt2 = f"resuma os seguintes textos em até 50 palavras cada:\n{textos_relevantes}"


        resposta = self.local_model.generate_response(prompt=prompt)
        #resumo
        #self.local_model.generate_response(prompt=prompt2)

        #f'{textos_relevantes}\n tamanho:{len(textos_relevantes)}\n indices: {indices}'
        return resposta


        

# Teste
if __name__ == "__main__":
    rag = RAGPipeline(max_tokens_per_request=2500, max_chunks=5, tokens_per_minute_limit=6000)
    query = "Concurso do ibama"
    print(rag.generate_full_answer(query))
    #print(rag.generate_answer(query))
