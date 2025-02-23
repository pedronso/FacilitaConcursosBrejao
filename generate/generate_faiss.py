import os
import sys
import pandas as pd
from vectorstore.faiss_store import FAISSVectorStore

BASE_DIR = "data/processed/configs"

import json

def criar_faiss_index():
    """Gera um índice FAISS para cada conjunto de chunks, verificando progresso e integridade."""
    pastas = sorted([f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))])

    for pasta in pastas:
        config_path = os.path.join(BASE_DIR, pasta)
        caminho_chunks = os.path.join(config_path, "chunks.csv")
        caminho_index = os.path.join(config_path, "faiss_index")
        checkpoint_file = os.path.join(config_path, "faiss_index_COMPLETED")
        metadata_file = os.path.join(config_path, "metadata.json")  # 🔹 Arquivo para salvar informações

        # 🔹 Se já foi concluído, pula
        if os.path.exists(checkpoint_file):
            print(f"🔹 {pasta} já concluído, pulando...")
            continue

        # ❌ Se não há chunks, pula
        if not os.path.exists(caminho_chunks):
            print(f"❌ Chunks não encontrados para {pasta}, pulando...")
            continue

        print(f"\n🚀 Processando FAISS para: {pasta}")

        df_chunks = pd.read_csv(caminho_chunks)
        chunks_list = df_chunks["Chunk"].dropna().tolist()
        total_chunks = len(chunks_list)

        print(f"📊 Total de embeddings a serem gerados: {total_chunks}")

        # ⚠️ Remove FAISS incompleto e recria
        if os.path.exists(caminho_index):
            print(f"⚠️ FAISS encontrado para {pasta} sem checkpoint. Recriando...")
            os.system(f"rm -rf {caminho_index}")

        # 🔹 Identifica o modelo de embedding pelo nome da pasta
        modelo_embedding = pasta.split("_")[1]  # 🔹 Supondo que o nome da pasta contenha o modelo

        # 🔹 Criando FAISS com progresso
        store = FAISSVectorStore(index_path=caminho_index, model_name=modelo_embedding)  # 🔹 Passa o modelo

        print(f"🔹 Utilizando modelo de embeddings: {modelo_embedding}")
        embeddings = []
        for i, chunk in enumerate(chunks_list):
            embeddings.append(store.embedding_model.get_embedding(chunk))
            if i % 100 == 0 or i == total_chunks - 1:  # Atualiza a cada 100 embeddings ou no final
                percent = ((i+1) / total_chunks) * 100
                print(f"\n[{i+1}/{total_chunks} ({percent:.2f}%)] 👍👍👍\n")


        # Criar índice FAISS
        store.create_index(embeddings)

        # 🔍 Salvar metadados do FAISS
        metadata = {
            "model_name": modelo_embedding,
            "total_chunks": total_chunks,
            "faiss_index_path": caminho_index
        }

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(f"📁 Metadados salvos em {metadata_file}")

        # 🔍 **Verificação de integridade**
        if verificar_faiss_integridade(caminho_index, total_chunks):
            # ✅ Renomeia o FAISS index para indicar conclusão
            caminho_index_completed = caminho_index + "_COMPLETED"
            os.rename(caminho_index, caminho_index_completed)
            print(f"✅ FAISS index renomeado para {caminho_index_completed} e marcado como COMPLETED.")


            # 🛑 Interrompe o script após processar um único arquivo
            sys.exit(0)
        else:
            print(f"❌ Erro ao verificar integridade do FAISS para {pasta}. Deletando índice corrompido e reiniciando...")
            os.system(f"rm -rf {caminho_index}")  # Remove índice corrompido para recriação
            sys.exit(1)  # Sai com erro para que o script reinicie

    print("✅ Todas as configurações já foram processadas.")


def verificar_faiss_integridade(caminho_index, total_chunks):
    """Verifica se o índice FAISS contém a quantidade esperada de embeddings."""
    try:
        store = FAISSVectorStore(index_path=caminho_index)
        embeddings_count = store.get_index_size()
        print(f"🔍 Verificação FAISS: {embeddings_count}/{total_chunks} embeddings encontrados.")
        return embeddings_count == total_chunks  # Retorna True se tudo estiver certo
    except Exception as e:
        print(f"❌ Erro ao verificar FAISS: {e}")
        return False  # Retorna False caso ocorra erro





def criar_faiss_index_unit(config_path):
    """
    Gera um índice FAISS apenas para uma configuração específica.

    Args:
        config_path (str): Caminho completo do diretório da configuração.
    """
    pasta = os.path.basename(config_path)
    caminho_chunks = os.path.join(config_path, "chunks.csv")
    caminho_index = os.path.join(config_path, "faiss_index")
    checkpoint_file = os.path.join(config_path, "faiss_index_COMPLETED")
    metadata_file = os.path.join(config_path, "metadata.json")  # 🔹 Arquivo para salvar informações

    # 🔹 Se já foi concluído, pula
    if os.path.exists(checkpoint_file):
        print(f"🔹 {pasta} já concluído, pulando...")
        return

    # ❌ Se não há chunks, pula
    if not os.path.exists(caminho_chunks):
        print(f"❌ Chunks não encontrados para {pasta}, pulando...")
        return

    print(f"\n🚀 Processando FAISS para: {pasta}")

    df_chunks = pd.read_csv(caminho_chunks)
    chunks_list = df_chunks["Chunk"].dropna().tolist()
    total_chunks = len(chunks_list)

    print(f"📊 Total de embeddings a serem gerados: {total_chunks}")

    # ⚠️ Remove FAISS incompleto e recria
    if os.path.exists(caminho_index):
        print(f"⚠️ FAISS encontrado para {pasta} sem checkpoint. Recriando...")
        os.system(f"rm -rf {caminho_index}")

    # 🔹 Identifica o modelo de embedding pelo nome da pasta
    modelo_embedding = pasta.split("_")[1]  # 🔹 Supondo que o nome da pasta contenha o modelo

    # 🔹 Criando FAISS com progresso
    store = FAISSVectorStore(index_path=caminho_index)

    print(f"🔹 Utilizando modelo de embeddings: {modelo_embedding}")
    embeddings = []
    for i, chunk in enumerate(chunks_list):
        embeddings.append(store.embedding_model.get_embedding(chunk))
        if i % 100 == 0 or i == total_chunks - 1:  # Atualiza a cada 100 embeddings ou no final
            percent = ((i+1) / total_chunks) * 100
            print(f"\n[{i+1}/{total_chunks} ({percent:.2f}%)] 👍👍👍\n")

    # Criar índice FAISS
    store.create_index(embeddings)

    # 🔍 Salvar metadados do FAISS
    metadata = {
        "model_name": modelo_embedding,
        "total_chunks": total_chunks,
        "faiss_index_path": caminho_index
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"📁 Metadados salvos em {metadata_file}")

    # 🔍 **Verificação de integridade**
    if verificar_faiss_integridade(caminho_index, total_chunks):
        # ✅ Renomeia o FAISS index para indicar conclusão
        caminho_index_completed = caminho_index + "_COMPLETED"
        os.rename(caminho_index, caminho_index_completed)
        print(f"✅ FAISS index renomeado para {caminho_index_completed} e marcado como COMPLETED.")
    else:
        print(f"❌ Erro ao verificar integridade do FAISS para {pasta}. Deletando índice corrompido e reiniciando...")
        os.system(f"rm -rf {caminho_index}")  # Remove índice corrompido para recriação
        sys.exit(1)  # Sai com erro para que o script reinicie

    print(f"✅ FAISS index concluído para {pasta}!")
    
if __name__ == "__main__":
    criar_faiss_index()
