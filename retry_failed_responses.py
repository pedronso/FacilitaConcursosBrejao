import json
import os
import time
from pipelines.rag import RAGPipeline  # Certifique-se de que esse import está correto

# Caminho do arquivo de respostas
RESULTS_FILE = "data/processed/respostas/DeepSeek_GTE-Large_200_40_ON_ON_ON_respostas.json"

def carregar_respostas_existentes():
    """Carrega as respostas já existentes e identifica quais precisam ser refeitas."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as file:
            respostas = json.load(file)
    else:
        respostas = {}

    # Identificar perguntas com erro
    perguntas_pendentes = [q for q, r in respostas.items() if "Erro ao gerar resposta" in r]

    return respostas, perguntas_pendentes

def gerar_respostas_pendentes(respostas, perguntas_pendentes):
    """Gera respostas apenas para as perguntas pendentes e atualiza o JSON."""
    if not perguntas_pendentes:
        print("✅ Todas as perguntas já foram respondidas corretamente.")
        return

    # Inicializa a pipeline RAG
    rag = RAGPipeline(config_name="DeepSeek_GTE-Large_200_40_ON_ON_ON")

    for pergunta in perguntas_pendentes:
        while True:  # Loop para garantir que a resposta seja gerada
            try:
                print(f"\n🔹 Tentando novamente: {pergunta}")
                resposta = rag.generate_answer(pergunta)

                if "list index out of range" in resposta:
                    print(f"⚠️ Erro de índice na resposta. Pulando pergunta: {pergunta}")
                    respostas[pergunta] = "Erro: Nenhum dado encontrado para essa pergunta."
                    break  # Sai do loop e continua com a próxima pergunta

                respostas[pergunta] = resposta  # Atualiza a resposta no dicionário
                print(f"✅ Resposta gerada: {resposta[:100]}...")  # Mostra um trecho da resposta
                break  # Sai do loop se a resposta for gerada com sucesso
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Erro ao gerar resposta para '{pergunta}': {error_msg}")

                if "rate_limit_exceeded" in error_msg:
                    wait_time = 30
                    print(f"⏳ Rate limit atingido. Aguardando {wait_time} segundos antes de tentar novamente...")
                    time.sleep(wait_time)
                    wait_time += 30  # Aumenta o tempo de espera progressivamente
                else:
                    respostas[pergunta] = f"Erro ao gerar resposta: {error_msg}"
                    break  # Sai do loop se for um erro diferente do rate limit

    # Salva o arquivo atualizado com as novas respostas
    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(respostas, file, indent=4, ensure_ascii=False)

    print("\n✅ Arquivo atualizado com as respostas corrigidas.")


if __name__ == "__main__":
    respostas, perguntas_pendentes = carregar_respostas_existentes()
    gerar_respostas_pendentes(respostas, perguntas_pendentes)
