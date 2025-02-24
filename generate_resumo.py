import os
import json
import numpy as np

# Diretório onde os arquivos de métricas estão armazenados
METRICS_DIR = "data/processed/metricas/"
METRICS_DIR_OUT = "data/processed/metricas/media/"
OUTPUT_FILE = os.path.join(METRICS_DIR_OUT, "metricas_resumo.json")

def calcular_medias_metricas():
    """Percorre os arquivos de métricas, calcula as médias e gera um JSON consolidado."""
    metricas_resumo = []

    # Garante que o diretório de saída existe
    os.makedirs(METRICS_DIR_OUT, exist_ok=True)

    arquivos_metricas = sorted([f for f in os.listdir(METRICS_DIR) if f.endswith("_metricas.json")])

    for idx, arquivo in enumerate(arquivos_metricas, start=1):
        caminho_arquivo = os.path.join(METRICS_DIR, arquivo)

        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            dados = json.load(f)

        # Extrai as avaliações
        avaliacoes = dados.get("avaliacoes", {})

        # Calcula a média
        if avaliacoes:
            media = np.mean(list(avaliacoes.values()))
        else:
            media = None  # Caso o arquivo não tenha avaliações

        # Adiciona ao resumo
        metricas_resumo.append({
            "ID": idx,
            "Configuracao": arquivo.replace("_metricas.json", ""),
            "Media": media
        })

    # Salva o JSON consolidado
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(metricas_resumo, f, indent=4, ensure_ascii=False)

    print(f"✅ Arquivo de resumo salvo em: {OUTPUT_FILE}")

if __name__ == "__main__":
    calcular_medias_metricas()
