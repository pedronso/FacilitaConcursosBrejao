import os
import pandas as pd
import time
from sklearn.metrics import accuracy_score, f1_score
from pipelines.rag import RAGPipeline

CSV_CHUNKS = "data/processed/results_extraction_chunks.csv"
METRICS_OUTPUT = "reports/metricas.csv"

def carregar_dados():
    """Carrega os chunks processados para teste."""
    if not os.path.exists(CSV_CHUNKS):
        print(f"❌ Arquivo {CSV_CHUNKS} não encontrado! Certifique-se de que a etapa de extração foi concluída.")
        return None
    
    df = pd.read_csv(CSV_CHUNKS)
    
    if "Chunk" not in df.columns:
        print(f"❌ A coluna 'Chunks' não está presente no arquivo {CSV_CHUNKS}.")
        return None
    
    return df

def avaliar_respostas(ground_truth, respostas_geradas):
    """Avalia as respostas geradas pelo RAG com base nas respostas esperadas."""
    if not ground_truth or not respostas_geradas:
        print("⚠️ Dados de teste insuficientes para avaliação.")
        return {}

    # estamos comparando as respostas diretamente 
    # TODO: (pode ser ajustado para NLP, 
    # ou outro llm analisar se as respostas condizem ou não, 
    # responder por exemplo unicamente com "y" ou "n, e fazermos esta contagem")
    accuracy = accuracy_score(ground_truth, respostas_geradas)
    f1 = f1_score(ground_truth, respostas_geradas, average="weighted")

    return {
        "Acurácia": accuracy,
        "F1-Score": f1
    }

def avaliar_sistema():
    """Executa a avaliação do desempenho do chatbot."""
    print("\n📊 Iniciando avaliação de métricas...")

    start_time = time.time()
    
    df_chunks = carregar_dados()
    if df_chunks is None:
        return

    rag = RAGPipeline()

    perguntas_teste = [
        "Quais são os concursos para engenheiros?",
        "Quais os prazos de inscrição?",
        "Tem algum concurso para nível médio?",
        "Quantas vagas estão abertas para o IBAMA?",
        "O MPU está com concursos abertos?"
    ]

    respostas_esperadas = [
        "Concursos para engenheiros incluem...",
        "Os prazos de inscrição variam, mas...",
        "Atualmente, existem concursos de nível médio como...",
        "O IBAMA tem 460 vagas abertas.",
        "Sim, o MPU anunciou vagas recentemente."
    ]

    respostas_geradas = []
    for pergunta in perguntas_teste:
        print(f"\n🔹 Pergunta: {pergunta}")
        resposta = rag.generate_answer(pergunta)
        print(f"💬 Resposta do modelo: {resposta}\n")
        respostas_geradas.append(resposta)

    # Avaliação das respostas
    metricas = avaliar_respostas(respostas_esperadas, respostas_geradas)

    # Criar DataFrame de métricas
    df_metricas = pd.DataFrame([metricas])
    
    # Salvar métricas em CSV
    os.makedirs(os.path.dirname(METRICS_OUTPUT), exist_ok=True)
    df_metricas.to_csv(METRICS_OUTPUT, index=False)

    print(f"✅ Métricas salvas em: {METRICS_OUTPUT}")

    total_time = time.time() - start_time
    print(f"\n⏳ Tempo total da avaliação: {total_time:.2f} segundos.")
