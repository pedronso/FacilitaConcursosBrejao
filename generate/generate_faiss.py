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

def criar_faiss_index_unit():
    """Gera um índice FAISS apenas para a N configuração na fila das pastas. (roda apenas uma unidade de faiss por vez)"""
    pastas = sorted([f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))])

    if len(pastas) < 2:
        print("❌ Não há uma segunda configuração para processar.")
        return

    # seleciona a segunda pasta
    pasta = pastas[1]  
    caminho_chunks = os.path.join(BASE_DIR, pasta, "chunks.csv")
    caminho_index = os.path.join(BASE_DIR, pasta, "faiss_index")

    if not os.path.exists(caminho_chunks):
        print(f"❌ Chunks não encontrados para {pasta}, pulando...")
        return

    print(f"\n🔍 Processando FAISS para: {pasta}")
    df_chunks = pd.read_csv(caminho_chunks)

    store = FAISSVectorStore(index_path=caminho_index)
    store.create_index(df_chunks["Chunk"].tolist())

    print(f"✅ FAISS index criado para {pasta}")
    
if __name__ == "__main__":
    criar_faiss_index()
