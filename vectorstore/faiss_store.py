import faiss
import numpy as np
import os
import pandas as pd
from models.embeddings_model import EmbeddingModel

class FAISSVectorStore:
    def __init__(self, index_path: str, model_name: str = None):
        """
        Inicializa a classe FAISS com um modelo de embeddings específico baseado na configuração da pasta.

        Args:
            index_path (str): Caminho do índice FAISS.
            model_name (str, opcional): Nome do modelo de embeddings a ser utilizado. 
                                        Se não fornecido, será extraído do nome da pasta.
        """
        self.index_path = index_path
        # Se model_name for fornecido, usa-o; caso contrário, extrai do nome da pasta
        if model_name:
            self.config_name = model_name
        else:
            self.config_name = self.get_config_name_from_path(index_path)
        # Extraímos o modelo real com base no nome da configuração
        self.model_name = self.extract_model_name(self.config_name)
        self.embedding_model = EmbeddingModel(model_name=self.model_name)
        self.dimension = 1024  # para E5-Large e GTE-Large
        self.index = None

        if os.path.exists(self.index_path):
            self.load_index()
        else:
            print(f"⚠️ FAISS index não encontrado para {self.config_name}. Criando novo...")
            self.create_empty_index()

    def get_config_name_from_path(self, index_path):
        """Extrai o nome da pasta que contém a configuração do FAISS."""
        return os.path.basename(os.path.dirname(index_path))

    def extract_model_name(self, config_name):
        """
        Extrai o nome do modelo de embeddings a partir do nome da pasta de configuração.
        
        Exemplo:
          - "DeepSeek_E5-Large_300_0_OFF_OFF_OFF" -> "intfloat/e5-large"
          - "DeepSeek_GTE-Large_200_40_ON_ON_ON" -> "thenlper/gte-large"
        """
        for model in ["E5-Large", "GTE-Large"]:
            if model in config_name:
                return f"thenlper/{model.lower()}" if "gte" in model.lower() else f"intfloat/{model.lower()}"
        print(f"⚠️ Modelo não identificado em '{config_name}'. Usando modelo padrão.")
        return "thenlper/gte-large"  # Modelo padrão

    def create_empty_index(self):
        """Cria um índice FAISS vazio com a dimensão correta."""
        self.index = faiss.IndexFlatL2(self.dimension)
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        self.save_index()

    def create_index(self, texts):
        """
        Cria um índice FAISS a partir de uma lista de textos ou embeddings.
        
        Se o primeiro elemento for uma string, assume que 'texts' é uma lista de textos
        e chama o método get_embedding para cada um; caso contrário, assume que 'texts'
        já contém os embeddings.
        """
        print(f"🔹 Criando FAISS index em: {self.index_path} usando modelo: {self.model_name}")
        if texts and isinstance(texts[0], str):
            embeddings = np.array(
                [self.embedding_model.get_embedding(text) for text in texts],
                dtype=np.float32
            )
        else:
            embeddings = np.array(texts, dtype=np.float32)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        self.save_index()
        print("✅ FAISS index criado e salvo.")

    def add_embedding(self, text):
        """Adiciona um embedding ao índice FAISS."""
        embedding = np.array([self.embedding_model.get_embedding(text)], dtype=np.float32)
        if self.index is None:
            print("⚠️ Índice FAISS não encontrado. Criando um novo...")
            self.create_empty_index()
        self.index.add(embedding)
        self.save_index()

    def save_index(self):
        """Salva o índice FAISS no caminho especificado."""
        faiss.write_index(self.index, self.index_path)

    def load_index(self):
        """Carrega um índice FAISS salvo."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"✅ FAISS index carregado para {self.config_name}. Modelo usado: {self.model_name}")
        else:
            print(f"❌ Erro: FAISS index não encontrado para {self.config_name}!")

    def get_index_size(self):
        """Retorna o número de embeddings no índice."""
        return self.index.ntotal if self.index else 0

    def search(self, query, k=5):
        """Faz busca no FAISS index."""
        query_embedding = np.array([self.embedding_model.get_embedding(query)], dtype=np.float32)
        _, indices = self.index.search(query_embedding, k)
        return indices[0]

if __name__ == "__main__":
    config_name = "DeepSeek_E5-Large_300_0_OFF_OFF_OFF"
    index_path = f"data/processed/configs/{config_name}/faiss_index_COMPLETED"
    store = FAISSVectorStore(index_path=index_path, model_name=config_name)
    store.create_index(["Texto exemplo para FAISS"])
    print(f"Tamanho do índice FAISS: {store.get_index_size()}")
