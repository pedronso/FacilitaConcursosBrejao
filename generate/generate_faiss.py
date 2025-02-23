import os
import pandas as pd
import json
from vectorstore.faiss_store import FAISSVectorStore

BASE_DIR = "data/processed/configs"

def criar_faiss_index():
    """Gera um índice FAISS para cada conjunto de chunks."""
    pastas = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]

    for pasta in pastas:
        caminho_chunks = os.path.join(BASE_DIR, pasta, "chunks.csv")
        caminho_index = os.path.join(BASE_DIR, pasta, "faiss_index")

        if os.path.exists(caminho_index):
            print(f"✅ FAISS index já existe para {pasta}, pulando...")
            continue

        if not os.path.exists(caminho_chunks):
            print(f"❌ Chunks não encontrados para {pasta}, pulando...")
            continue

        df_chunks = pd.read_csv(caminho_chunks)
        store = FAISSVectorStore(index_path=caminho_index)
        store.create_index(df_chunks["Chunk"].tolist())

        print(f"✅ FAISS index criado para {pasta}")

if __name__ == "__main__":
    criar_faiss_index()
