import os
import json
import numpy as np

# Diretório onde os arquivos de tempo estão armazenados
TIME_DIR = "data/processed/time_results/"
TIME_DIR_OUT = "data/processed/metricas/media/"
OUTPUT_FILE = os.path.join(TIME_DIR_OUT, "tempos_resumo.json")

# Criar diretório de saída, se não existir
os.makedirs(TIME_DIR_OUT, exist_ok=True)

def calcular_medias_tempo():
    """Percorre os arquivos de tempo, calcula as médias e gera um JSON consolidado."""
    tempos_resumo = []
    tempos_deepseek = []
    tempos_llama3 = []

    arquivos_tempo = sorted([f for f in os.listdir(TIME_DIR) if f.endswith("_time_results.json")])

    for idx, arquivo in enumerate(arquivos_tempo, start=1):
        caminho_arquivo = os.path.join(TIME_DIR, arquivo)

        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            dados = json.load(f)

        # Extrai os tempos individuais
        tempos = dados.get("individual_times", {})

        # Calcula a média
        if tempos:
            media = np.mean(list(tempos.values()))
        else:
            media = None  # Caso o arquivo não tenha tempos

        # Determina se é DeepSeek ou LLaMA-3
        if "DeepSeek" in arquivo:
            tempos_deepseek.append(media)
        elif "LLaMA-3" in arquivo:
            tempos_llama3.append(media)

        # Adiciona ao resumo
        tempos_resumo.append({
            "ID": idx,
            "Configuracao": arquivo.replace("_time_results.json", ""),
            "Media_Tempo": media
        })

    # Garantindo que a média de cada modelo seja feita corretamente
    media_deepseek = np.sum(tempos_deepseek) / 12 if tempos_deepseek else None
    media_llama3 = np.sum(tempos_llama3) / 2 if tempos_llama3 else None

    resumo_final = {
        "Tempos_Detalhados": tempos_resumo,
        "Media_Geral": {
            "DeepSeek": media_deepseek,
            "LLaMA-3": media_llama3
        }
    }

    # Salva o JSON consolidado
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(resumo_final, f, indent=4, ensure_ascii=False)

    print(f"✅ Arquivo de resumo salvo em: {OUTPUT_FILE}")

if __name__ == "__main__":
    calcular_medias_tempo()
