from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Nova importação correta
import tests_vars
#from langchain_ollama import OllamaEmbeddings
#import ollama

class EmbeddingModel:
    def __init__(self, model_name=None):
        """Inicializa o modelo de embeddings do Hugging Face."""
        # Obtém o modelo do arquivo tests_vars, se não estiver definido, usa E5-Large como padrão
        self.model_name = model_name or tests_vars.dict_models.get('embedding_model', 'intfloat/e5-large-v2')

        try:
            print(f"🔹 Carregando modelo de embeddings: {self.model_name}")
            self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)
            self.embedding_count = 1
        except Exception as e:
            print(f"❌ Erro ao carregar modelo de embeddings {self.model_name}: {e}")
            raise

    def get_embedding(self, text):
        """Gera embeddings para um determinado texto."""
        print(f'👍 Gerando embeddings... count: {self.embedding_count}')
        self.embedding_count += 1
        return self.embedding_model.embed_query(text)

"""
class OllamaEmbeddingModel:
    def __init__(self, model_name= 'mxbai-embed-large:latest'):
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        
    def get_embedding(self, text):
        print(f'👍 gerando embeddings...')
        
        #response = ollama.embed(model="mxbai-embed-large:latest", input=text)
        #embeddings = response["embeddings"]
        #return embeddings
        
        a = self.embedding_model.embed_query(text)
        #print(a)
        return a
"""
    
# Teste
if __name__ == "__main__":
    model = EmbeddingModel()
    texto_teste = "Este é um teste para gerar embeddings."
    embedding = model.get_embedding(texto_teste)
    print(f"✅ Embedding gerado: {embedding}...")
