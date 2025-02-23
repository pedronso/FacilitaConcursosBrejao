import os
import pandas as pd
import json
from pipelines.extractor import chunking_texto

BASE_DIR = "data/processed/configs"
TEXTOS_PATH = "data/extracted_pedro"

def processar_chunks():
    """Cria chunks para cada configuração armazenada."""
    pastas = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]

    for pasta in pastas:
        caminho_config = os.path.join(BASE_DIR, pasta, "config.json")
        caminho_chunks = os.path.join(BASE_DIR, pasta, "chunks.csv")

        print(f"🔍 Configuração: {pasta}")
        print(f"   📄 Caminho do JSON: {caminho_config}")
        print(f"   📄 Caminho dos Chunks: {caminho_chunks}")
        print(f"   📁 Caminho FAISS: {os.path.join(BASE_DIR, pasta, 'faiss_index')}")

        if os.path.exists(caminho_chunks):
            print(f"✅ Chunks já existem para {pasta}, pulando...")
            continue

        with open(caminho_config, "r") as file:
            config = json.load(file)

        arquivos = [os.path.join(TEXTOS_PATH, f) for f in os.listdir(TEXTOS_PATH) if f.endswith(".txt")]
        all_chunks = []

        for arquivo in arquivos:
            chunks = chunking_texto(
                arquivo,
                labeled=config["Label"] == "ON",
                normalized=config["Normalization"] == "ON",
                remove_stopwords=config["Stopwords"] == "ON",
                chunk_size=config["Chunk"],
                chunk_overlap=config["Overlap"]
            )
            all_chunks.extend(chunks)

        df_chunks = pd.DataFrame({"Chunk": all_chunks})
        df_chunks.to_csv(caminho_chunks, index=False)

        print(f"✅ Chunks gerados para {pasta}")

if __name__ == "__main__":
    processar_chunks()
