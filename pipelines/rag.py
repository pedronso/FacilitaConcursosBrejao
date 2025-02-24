import os
import time
import math
import pandas as pd
from vectorstore.faiss_store import FAISSVectorStore
from models.llm_model import LLMModel, LocalLLMModel
import tests_vars
from tests_vars import idx

class RAGPipeline:
    def __init__(self, config_name, base_dir="data/processed/configs",
                 max_tokens_per_request=2500, max_chunks=tests_vars.dict_models["topk"], tokens_per_minute_limit=6000):
        """
        Inicializa a pipeline RAG com caminhos específicos para cada configuração.
        - config_name: Nome da configuração que está sendo carregada.
        """
        self.config_name = config_name
        config_path = os.path.join(base_dir, config_name)

        self.faiss_index_path = os.path.join(config_path, "faiss_index_COMPLETED")
        self.chunks_csv_path = os.path.join(config_path, "chunks.csv")

        # Verificações para garantir que os arquivos necessários existem
        if not os.path.exists(self.faiss_index_path):
            raise FileNotFoundError(f"❌ FAISS index não encontrado para {config_name}: {self.faiss_index_path}")

        if not os.path.exists(self.chunks_csv_path):
            raise FileNotFoundError(f"❌ Arquivo chunks.csv não encontrado para {config_name}: {self.chunks_csv_path}")

        print(f"🔹 Carregando FAISS index de: {self.faiss_index_path}")
        print(f"🔹 Carregando chunks de: {self.chunks_csv_path}")

        # Modelos
        self.local_model = LocalLLMModel()
        self.vector_store = FAISSVectorStore(index_path=self.faiss_index_path)
        self.vector_store.load_index()
        self.llm = LLMModel()

        # Carregar os chunks
        self.df_original = pd.read_csv(self.chunks_csv_path)
        self.df_chunks = self.df_original.dropna().reset_index(drop=True)

        self.max_tokens_per_request = max_tokens_per_request
        self.max_chunks = max_chunks
        self.tokens_per_minute_limit = tokens_per_minute_limit
        self.tokens_used = 0
        self.start_time = time.time()
        
    def wait_if_needed(self, error_message):
        """
        Aguarda um tempo progressivo caso ocorra erro de rate limit.
        - O primeiro erro causa um sleep de 30s.
        - Cada novo erro aumenta o tempo de espera em +30s.
        """
        wait_time = 30
        while "rate_limit_exceeded" in error_message:
            print(f"⏳ Rate limit atingido. Aguardando {wait_time} segundos antes de tentar novamente...")
            time.sleep(wait_time)
            wait_time += 30  # Aumenta o tempo de espera progressivamente
            
    def generate_answer(self, query):
        """Busca os chunks relevantes e gera resposta via LLM."""
        
        # Normaliza a query para garantir que contenha a sigla correta do concurso
        query_corrigida = self.normalize_query(query)

        if "Especifique um concurso" in query_corrigida:
            return query_corrigida

        # Busca os índices relevantes no FAISS
        indices = self.buscar_indices_no_faiss(query_corrigida)
        
        # Filtra os índices pelo concurso correspondente

        indices_filtrados = self.filtrar_indices_por_concurso(indices, query_corrigida)
        
        # Obtém os textos correspondentes aos índices filtrados
        textos_relevantes = self.obter_textos_relevantes(indices_filtrados)

        # Definindo um valor padrão para evitar erro de variável não definida
        resposta = "Não foi possível gerar uma resposta no momento."

        while True:  # loop infinito para garantir que TODAS as respostas dos casos de teste sejam geradas.
            try:
                resposta = self.gerar_resposta_com_llm(query_corrigida, textos_relevantes)
                break  # Sai do loop se a resposta for gerada com sucesso
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Erro ao gerar resposta para '{query}': {error_msg}")
                if "rate_limit_exceeded" in error_msg:
                    self.wait_if_needed(error_msg)  # aguardar 30sec + 30 + 30, antes de tentar novamente
                else:
                    resposta = f"Erro ao gerar resposta: {error_msg}"
                    break  # Sai do loop se for um erro diferente do rate limit

        return resposta

    def normalize_query(self, query):
        """Garante que a query contenha a sigla correta do concurso."""
        concursos_mapeados = {
            "ibge": ["ibge", "instituto brasileiro de geografia e estatística"],
            "cnen": ["cnen", "comissão nacional de energia nuclear"],
            "cceb": ["cceb", "censo cidades estudantil brasil"],
            "aeronautica": ["aeronautica", "força aérea", "fab", "aeronáutica"],
            "aeb": ["aeb", "agência espacial brasileira"],
            "ibama": ["ibama", "instituto brasileiro do meio ambiente"],
            "funai": ["funai", "fundação nacional do índio"],
            "trf": ["trf", "tribunal regional federal"],
            "marinha": ["marinha", "força naval", "navy"],
            "mpu": ["mpu", "ministério público da união", "ministerio publico da uniao"]
        }
    
        query = query.lower()
        concurso_encontrado = None

        for sigla, palavras in concursos_mapeados.items():
            for palavra in palavras:
                if palavra in query:
                    concurso_encontrado = sigla
                    break
            if concurso_encontrado:
                break

        if not concurso_encontrado:
            return "Especifique um concurso na sua pergunta. Concursos disponíveis: " + ", ".join(concursos_mapeados.keys())

        return f"{query} ({concurso_encontrado})"

    def buscar_indices_no_faiss(self, query):
        """Realiza a busca no FAISS e retorna os índices encontrados."""
        query = query.strip().lower()
        indices = self.vector_store.search(query, 100)
        return [int(i) for i in indices if 0 <= i < len(self.df_chunks)]

    def filtrar_indices_por_concurso(self, indices, query):
        """Filtra os índices com base no concurso mencionado na query."""
        concursos_limites = {
            'aeb': (idx[0] - 1, idx[1]),
            'aeronautica': (idx[1] + 1, idx[2]),
            'cceb': (idx[2] + 1, idx[3]),
            'cnen': (idx[3] + 1, idx[4]),
            'funai': (idx[4] + 1, idx[5]),
            'ibama': (idx[5] + 1, idx[6]),
            'ibge': (idx[6] + 1, idx[7]),
            'marinha': (idx[7] + 1, idx[8]),
            'mpu': (idx[8] + 1, idx[9]),
            'trf': (idx[9] + 1, 999999)
        }
        print(concursos_limites)

        for concurso, (min_idx, max_idx) in concursos_limites.items():
            if concurso in query:
                indices = [i for i in indices if min_idx <= i <= max_idx]

        return indices[:self.max_chunks]

    def obter_textos_relevantes(self, indices):
        """Obtém os textos correspondentes aos índices filtrados."""
        return " ".join([self.df_chunks.iloc[i]["Chunk"] for i in indices])

    def gerar_resposta_com_llm(self, query, textos_relevantes):
        """Gera a resposta utilizando o LLM com base nos textos filtrados."""
        print(f"🔍 Processando pergunta: {query}")
        print(f"🔹 Tokens nos textos selecionados: {len(textos_relevantes.split())}")

        prompt = f"Baseando-se somente nos seguintes textos:{textos_relevantes}\n\n Responda: {query}"
        return self.llm.generate_response(prompt)


if __name__ == "__main__":
    config_name = "LLaMA-3_E5-Large_300_0_OFF_OFF_OFF"
    rag = RAGPipeline(config_name)
    query = "Concurso do ibama"
    print(rag.generate_answer(query))
