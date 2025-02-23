import os
import json

BASE_DIR = "data/processed/configs"

def verificar_modelos_de_embeddings():
    """Verifica os modelos de embeddings utilizados para criar os índices FAISS e compara com o esperado."""

    pastas = sorted([f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))])
    erros = []

    for pasta in pastas:
        config_path = os.path.join(BASE_DIR, pasta)
        metadata_file = os.path.join(config_path, "metadata.json")

        # Verifica se o metadata.json existe
        if not os.path.exists(metadata_file):
            print(f"⚠️ {pasta} não possui metadata.json, impossível verificar modelo de embedding.")
            erros.append(f"{pasta}: metadata.json ausente")
            continue

        # Lê os metadados do FAISS index
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        modelo_registrado = metadata.get("model_name", "N/A")

        # Extrai o modelo do nome da pasta
        modelo_esperado = "E5-Large" if "E5-Large" in pasta else "GTE-Large"

        # Compara os modelos
        if modelo_registrado != modelo_esperado:
            print(f"❌ ERRO: Modelo divergente em {pasta} -> Esperado: {modelo_esperado} | Registrado: {modelo_registrado}")
            erros.append(f"{pasta}: Esperado {modelo_esperado}, mas encontrado {modelo_registrado}")
        else:
            print(f"✅ OK: {pasta} -> Modelo correto ({modelo_registrado})")

    # Relatório final
    print("\n🔍 Revisão completa!")
    if erros:
        print("\n🚨 INCONSISTÊNCIAS DETECTADAS:")
        for erro in erros:
            print(f"   - {erro}")
    else:
        print("✅ Todos os modelos estão corretos!")

if __name__ == "__main__":
    verificar_modelos_de_embeddings()
