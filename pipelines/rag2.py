from multiprocessing import process
import time
import math
from vectorstore.faiss_store import FAISSVectorStore
from models.llm_model import LLMModel
from models.llm_model import LocalLLMModel
import pandas as pd
import tests_vars

class RAGPipeline:
    def __init__(self, faiss_index_path="data/embeddings/faiss_index_1", 
                 chunks_csv_path="data/processed/results_extraction_chunks.csv",
                 max_tokens_per_request=2500, max_chunks=tests_vars.dict_models["topk"], tokens_per_minute_limit=6000):
        """
        Inicializa a pipeline RAG.
        - max_tokens_per_request: número máximo de tokens enviados para o LLM por requisição.
        - max_chunks: número máximo de chunks recuperados do FAISS.
        - tokens_per_minute_limit: limite de tokens por minuto imposto pela API.
        """
        self.local_model = LocalLLMModel()
        self.vector_store = FAISSVectorStore(index_path=faiss_index_path) #atualizar caminhos output de acordo com as novas configs
        self.vector_store.load_index()
        
        self.llm = LLMModel()
        
        self.df_original = pd.read_csv(chunks_csv_path)
        self.df_chunks = self.df_original.dropna().reset_index(drop=True)
        #self.df_chunks = self.df_original.melt(var_name="Chunk_Index", value_name="Chunk").dropna().reset_index(drop=True)

        self.max_tokens_per_request = max_tokens_per_request
        self.max_chunks = max_chunks
        self.tokens_per_minute_limit = tokens_per_minute_limit
        self.tokens_used = 0 
        self.start_time = time.time()   
        self.concursos_mapeados = {
            "ibge": ["ibge", "instituto brasileiro de geografia e estatística"],
            "cnen": ["cnen", "comissão nacional de energia nuclear"],
            "cceb": ["cceb", "censo cidades estudantil brasil"],
            "aeronautica": ["aeronautica", "força aérea", "fab", "aeronáutica"],
            "aeb": ["aeb", "agência espacial brasileira"],
            "ibama": ["ibama", "instituto brasileiro do meio ambiente"],
            "funai": ["funai", "fundação nacional do índio"],
            "trf": ["trf", "tribunal regional federal"],
            "marinha": ["marinha", "força naval", "navy"],
        }
        
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
    
    def corrigir_concurso(self, query):
        pass
    #    """Corrige possíveis erros de digitação e garante que a sigla correta seja usada."""
    #    palavras_usuario = query.lower().split()
    #    concursos_validos = list(self.concursos_mapeados.keys())  # Lista de siglas oficiais
    #    
    #    for i, palavra in enumerate(palavras_usuario):
    #        match, score = process.extractOne(palavra, concursos_validos, score_cutoff=80)  # Encontra similaridade acima de 80%
    #        if match:
    #            palavras_usuario[i] = match  # Substitui palavra errada pela sigla correta
    #
    #    return " ".join(palavras_usuario)  # Retorna a query corrigida
    
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
        
        # 1️⃣ Corrigir e normalizar a query
        #query_corrigida = self.corrigir_concurso(query)
        query_corrigida = self.normalize_query(query)
        #print(f"🔍 Query corrigida: {query_corrigida}")  # Debug para verificar correção

        if "Especifique um concurso" in query_corrigida:
            return query_corrigida

        # 2️⃣ Buscar índices relevantes no FAISS
        indices = self.buscar_indices_no_faiss(query_corrigida)
        
        # 3️⃣ Filtrar índices pelo concurso correspondente
        indices_filtrados = self.filtrar_indices_por_concurso(indices, query_corrigida)
        
        # 4️⃣ Selecionar os textos correspondentes aos índices
        textos_relevantes = self.obter_textos_relevantes(indices_filtrados)

        # 5️⃣ Gerar a resposta usando o LLM
        resposta = self.gerar_resposta_com_llm(query_corrigida, textos_relevantes)

        return resposta


    def buscar_indices_no_faiss(self, query):
        """Realiza a busca no FAISS e retorna os índices encontrados."""
        query = query.strip().lower()
        indices = self.vector_store.search(query, 50)
        return [int(i) for i in indices if 0 <= i < len(self.df_chunks)]


    def filtrar_indices_por_concurso(self, indices, query):
        """Filtra os índices com base no concurso mencionado na query."""
        if 'ibge' in query:
            indices = [i for i in indices if i <= 84]
        elif 'cnen' in query:
            indices = [i for i in indices if 84 < i <= 608]
        elif 'cceb' in query:
            indices = [i for i in indices if 608 < i <= 776]
        elif 'aeronautica' in query:
            indices = [i for i in indices if 776 < i <= 1118]
        elif 'aeb' in query:
            indices = [i for i in indices if 1118 < i <= 1431]
        elif 'ibama' in query:
            indices = [i for i in indices if 1431 < i <= 1819]
        elif 'funai' in query:
            indices = [i for i in indices if 1819 < i <= 2065]
        elif 'trf' in query:
            indices = [i for i in indices if 2065 < i <= 2462]
        elif 'marinha' in query:
            indices = [i for i in indices if 2462 < i <= 2821]
        
        return indices[:self.max_chunks]


    def obter_textos_relevantes(self, indices):
        """Obtém os textos correspondentes aos índices filtrados."""
        return " ".join([self.df_chunks.iloc[i]["Chunk"] for i in indices])


    def gerar_resposta_com_llm(self, query, textos_relevantes):
        """Gera a resposta utilizando o LLM com base nos textos filtrados."""
        print(f"🔍 Processando pergunta: {query}")
        print(f"🔹 Índices selecionados: {len(textos_relevantes.split())} tokens")

        prompt = f"Baseando-se somente nos seguintes textos:{textos_relevantes}\n\n Responda: {query}"
        return self.llm.generate_response(prompt)

    
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
        
        if 'ibge' in query.strip().lower():
            print("aiai bolsonaro é bom de mais")
            indices = [i for i in indices if i <= 104]
        
        print(f'Pegando índices: {indices} query {query}')

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
