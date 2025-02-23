import os
import json
import pandas as pd

BASE_DIR = "data/processed/configs"

def atualizar_metadados():
    """Atualiza os metadados (`metadata.json`) para todas as configurações existentes, garantindo a contagem correta de chunks."""
    
    pastas = sorted([f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))])

    for pasta in pastas:
        config_path = os.path.join(BASE_DIR, pasta)
        caminho_chunks = os.path.join(config_path, "chunks.csv")
        caminho_faiss = os.path.join(config_path, "faiss_index_COMPLETED")
        metadata_file = os.path.join(config_path, "metadata.json")

        # Verifica se os arquivos necessários existem
        if not os.path.exists(caminho_chunks) or not os.path.exists(caminho_faiss):
            print(f"⚠️ Pulando {pasta}: Chunks ou FAISS index não encontrados.")
            continue

        # Lê os chunks para contar a quantidade real
        df_chunks = pd.read_csv(caminho_chunks)
        total_chunks = df_chunks.shape[0]  # Conta o número de linhas

        # Identifica o modelo de embedding pelo nome da pasta
        modelo_embedding = "E5-Large" if "E5-Large" in pasta else "GTE-Large"

        # Criar/atualizar metadados
        metadata = {
            "model_name": modelo_embedding,
            "total_chunks": total_chunks,
            "faiss_index_path": caminho_faiss
        }

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(f"✅ Metadados atualizados para {pasta}: {metadata_file}")

if __name__ == "__main__":
    atualizar_metadados()
