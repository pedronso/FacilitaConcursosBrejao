from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Nova importação correta
#from langchain_ollama import OllamaEmbeddings
#import ollama

class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Inicializa o modelo de embeddings do Hugging Face."""
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def get_embedding(self, text):
        """Gera embeddings para um determinado texto."""
        print(f'👍 gerando embeddings...')
        embedded_text = self.embedding_model.embed_query(text)
        #print(a)
        return embedded_text

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
    print(f"Embedding gerado: {embedding}...")
