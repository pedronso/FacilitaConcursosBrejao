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
        self.df_chunks = self.df_original[['Chunk']].dropna().reset_index(drop=True)
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
            time.sleep(30)
            self.reset_token_usage_if_needed()

    def split_text(self, text, max_length):
        """Divide um texto em partes que não excedam `max_length` tokens."""
        words = text.split()
        num_parts = math.ceil(len(words) / max_length)
        return [" ".join(words[i * max_length:(i + 1) * max_length]) for i in range(num_parts)]

    def generate_answer(self, query):
        """Busca os chunks relevantes e gera resposta via LLM em partes para evitar perda de dados."""
        print(f"🔍 Total de Chunks no CSV: {len(self.df_chunks)}")
        print(f"🔍 Exemplo de Chunk: {self.df_chunks.iloc[0]['Chunk'] if len(self.df_chunks) > 0 else 'Nenhum chunk encontrado'}")
        
        indices = self.vector_store.search(query, k=self.max_chunks)
        indices = indices[0]  # Garantindo que pegamos apenas a lista interna

        # Verifica se os índices são válidos antes de acessar
        valid_indices = [int(i) for i in indices if isinstance(i, (int, float)) and 0 <= i < len(self.df_chunks)]

        # Exibir quais chunks estão sendo utilizados
        print(f"🔍 Chunks recuperados: {valid_indices}")

        # Extração dos chunks correspondentes
        textos_relevantes = "\n\n".join(self.df_chunks.iloc[i]["Chunk"] for i in valid_indices)

        print(f"📝 Textos extraídos para resposta:\n{textos_relevantes[:500]}")  # Mostra parte do conteúdo extraído
        prompt = f"Baseando-se SOMENTE nos seguintes trechos extraídos de documentos oficiais:\n\n{textos_relevantes}\n\nResponda a pergunta da forma mais objetiva possível: {query}"

        try:
            resposta = self.llm.generate_response(prompt)
        except Exception as e:
            import traceback
            print(f"❌ Erro ao gerar resposta para '{query}': {e}")
            traceback.print_exc()
            return "❌ Erro ao processar a resposta."

        return resposta
    
    def generate_full_answer(self, query):
        """
        print("pegando indices...")
        indices = self.vector_store.search(query, k=self.max_chunks)
        indices = [int(i) for i in indices if 0 <= i < len(self.df_chunks)]
        print(f'pegado {indices}')

        textos_relevantes = " ".join([self.df_chunks.iloc[i]["Chunk"] for i in indices])
        """
        
        print(f"🔍 Total de Chunks no CSV: {len(self.df_chunks)}")
        print(f"🔍 Exemplo de Chunk: {self.df_chunks.iloc[0]['Chunk'] if len(self.df_chunks) > 0 else 'Nenhum chunk encontrado'}")
        
        indices = self.vector_store.search(query, k=self.max_chunks)
        indices = [int(i) for i in indices[0] if isinstance(i, (int, float)) and 0 <= i < len(self.df_chunks)]

        if not indices:
            return "❌ Nenhum chunk relevante encontrado."

        for idx in indices:
            print(f"🔹 Chunk {idx}: {self.df_chunks.iloc[idx]['Chunk']}")


        textos_relevantes = " ".join([self.df_chunks.iloc[i]["Chunk"] for i in indices])
        print(f"🔹 Chunks recuperados: {textos_relevantes}")
        """
        textos_relevantes = " ".join([
            " ".join(map(str, self.df_chunks.iloc[i].dropna())) for i in indices
        ])
        """

        #print(f"Gerando resposta com texto inteiro...{textos_relevantes}")

        #prompt = f"Baseando-se somente nos seguintes textos:\n{textos_relevantes}\n\n responda: {query}"
        prompt = f"Baseando-se SOMENTE nos seguintes trechos retirados de documentos oficiais:\n\n{textos_relevantes}\n\nResponda da forma mais objetiva possível à seguinte pergunta: {query}"

        #prompt = f"Me diga o que tem nos seguintes textos separando cada um:\n{textos_relevantes}"
        #prompt2 = f"resuma os seguintes textos em até 50 palavras cada:\n{textos_relevantes}"


        resposta = self.local_model.generate_response(prompt=prompt)
        #resumo
        #self.local_model.generate_response(prompt=prompt2)

        #f'{textos_relevantes}\n tamanho:{len(textos_relevantes)}\n indices: {indices}'
        return resposta


        

# Teste
if __name__ == "__main__":
    rag = RAGPipeline(max_tokens_per_request=2500, max_chunks=5, tokens_per_minute_limit=6000)
    query = "Concurso do IBAMA"
    print(rag.generate_full_answer(query))
    #print(rag.generate_answer(query))
