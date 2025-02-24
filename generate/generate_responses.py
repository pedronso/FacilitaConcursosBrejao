import os
import json
import time
import pandas as pd
from pipelines.rag import RAGPipeline
import tests_vars

CONFIGS_DIR = "data/processed/configs"
RESULTS_DIR = "data/processed/respostas"
TIME_RESULTS_DIR = "data/processed/time_results"
MODELS_USED_DIR = "data/processed/modelos_usados"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TIME_RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_USED_DIR, exist_ok=True)

perguntas = [
    # IBGE
    "Quais são as funções disponíveis no processo seletivo do IBGE e quais os requisitos para cada uma?",
    "Qual o prazo para inscrições e a taxa de inscrição para o processo seletivo do IBGE?",
    "Como será o processo de avaliação e classificação dos candidatos no concurso do IBGE?",
    "Qual o valor da remuneração e dos auxílios oferecidos para os cargos disponíveis no IBGE?",

    # Marinha
    "Quais são os requisitos de idade e escolaridade para ingresso na marinha?",
    "Como será composta a Prova Escrita Objetiva e quais os critérios de aprovação para o concurso da marinha?",
    "Como funciona o Teste de Aptidão Física de Ingresso (TAF-i) para os candidatos no concurso da marinha?",
    "Quais são as cidades onde serão realizadas as provas e eventos complementares do concurso da marinha?",

    # MPU
    "Quais os cargos disponíveis no concurso do MPU e quantas vagas são oferecidas?",
    "Como será composta a prova objetiva e quais são os critérios de classificação e eliminação do concurso do mpu?",
    "Quais são os requisitos mínimos para investidura nos cargos do MPU?",
    "Qual o prazo de validade do concurso do mpu e se pode ser prorrogado?",

    # TRF
    "Quantas vagas estão disponíveis para o cargo de Juiz Federal Substituto no TRF?",
    "Como será estruturada a prova oral e quais os critérios de avaliação para o concurso do TRF?",
    "Quais são os requisitos mínimos exigidos para concorrer ao cargo de Juiz Federal Substituto do TRF?",
    "Quais etapas compõem o concurso do trf e como funciona o sistema de classificação?",

    # AEB
    "Quais os cargos oferecidos no concurso da AEB e quais os requisitos para cada um?",
    "Como será composta a prova objetiva e quais são os critérios de eliminação da AEB?",
    "Quais cidades aplicarão as provas do concurso da AEB?",
    "Qual é a jornada de trabalho e o valor da remuneração dos cargos da AEB?",

    # Aeronáutica
    "Qual a idade máxima permitida para ingresso no Curso de Formação de Sargentos da Aeronáutica?",
    "Como funciona o processo de escolha da especialidade dentro do curso da aeronáutica?",
    "Quais são os critérios de aprovação no Teste de Aptidão Física aeronáutica?",
    "Quais são as etapas do concurso para ingresso na Aeronáutica?",

    # CCEB
    "Quais são os cargos oferecidos no concurso do CCEB e quais os requisitos mínimos?",
    "Como será composta a prova objetiva e qual o critério de classificação do CCEB?",
    "Quais estados terão vagas disponíveis para os cargos do CCEB?",
    "Qual a duração do contrato para os aprovados no concurso do CCEB?",

    # FUNAI
    "Quais funções estão disponíveis no concurso da FUNAI e quais os requisitos para cada uma?",
    "Como será o processo de avaliação e classificação dos candidatos no concurso da FUNAI?",
    "Qual é o valor da remuneração e os benefícios oferecidos para os cargos da FUNAI?",
    "Onde serão realizadas as provas do concurso da FUNAI?",

    # IBAMA
    "Quais são os cargos oferecidos no concurso do IBAMA e quantas vagas estão disponíveis?",
    "Como será composta a prova objetiva e quais são os critérios de classificação do concurso do IBAMA?",
    "Qual é a jornada de trabalho e a remuneração inicial para os cargos do IBAMA?",
    "Como será realizada a lotação dos aprovados no concurso do IBAMA?",
]


def gerar_respostas():
    """Executa a RAG para todas as configurações, salva as respostas e os tempos de execução."""
    if not os.path.exists(CONFIGS_DIR):
        print(f"❌ Diretório de configurações não encontrado: {CONFIGS_DIR}")
        return

    for config_name in sorted(os.listdir(CONFIGS_DIR)): 
        config_path = os.path.join(CONFIGS_DIR, config_name)
        chunks_path = os.path.join(config_path, "chunks.csv")
        faiss_path = os.path.join(config_path, "faiss_index_COMPLETED")
        results_filename = os.path.join(RESULTS_DIR, f"{config_name}_respostas.json")
        time_filename = os.path.join(TIME_RESULTS_DIR, f"{config_name}_time_results.json")
        models_filename = os.path.join(MODELS_USED_DIR, f"{config_name}_modelos_usados.json")

        # Se já há respostas para essa configuração, pula
        if os.path.exists(results_filename):
            print(f"🔹 Respostas para {config_name} já geradas, pulando...")
            continue

        # Verificar se os arquivos necessários existem
        if not os.path.exists(chunks_path) or not os.path.exists(faiss_path):
            print(f"⚠️ Ignorando configuração incompleta: {config_name}")
            continue

        tests_vars.process_indexes(config_name)
        print(f"\n🚀 Testando configuração: {config_name}")

        start_time = time.time()

        try:
            rag = RAGPipeline(config_name)
        except FileNotFoundError as e:
            print(f"❌ Erro ao carregar configuração {config_name}: {e}")
            continue

        perguntas_respostas_dict = {}
        question_times = {}  # Tempo para cada pergunta
        models_used = {}     # Modelo utilizado para cada pergunta

        for i, pergunta in enumerate(perguntas, 1):
            print(f"\n🔹 [{i}/{len(perguntas)}] Pergunta: {pergunta}")
            t0 = time.time()
            try:
                resposta = rag.generate_answer(pergunta)
                perguntas_respostas_dict[pergunta] = str(resposta)
                # Extrai o modelo usado do prefixo da resposta
                if resposta.startswith("[Model used:"):
                    end_index = resposta.find("]")
                    modelo = resposta[len("[Model used: "):end_index]
                    models_used[pergunta] = modelo
                else:
                    models_used[pergunta] = "Modelo desconhecido"
                print(f"💬 Resposta: {resposta}")
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Erro ao gerar resposta para '{pergunta}': {error_msg}")
                perguntas_respostas_dict[pergunta] = f"Erro ao gerar resposta: {error_msg}"
                models_used[pergunta] = "Erro"
            t1 = time.time()
            elapsed = t1 - t0
            question_times[pergunta] = elapsed
            print(f"⏱ Tempo para responder: {elapsed:.2f} segundos")

        total_time_questions = sum(question_times.values())
        average_time = total_time_questions / len(question_times) if question_times else 0
        time_results = {
            "individual_times": question_times,
            "total_time": total_time_questions,
            "average_time": average_time
        }

        # Salvar respostas
        with open(results_filename, "w", encoding="utf-8") as f:
            json.dump(perguntas_respostas_dict, f, indent=4, ensure_ascii=False)
        print(f"📁 Respostas salvas em: {results_filename}")

        # Salvar tempos de execução
        with open(time_filename, "w", encoding="utf-8") as f:
            json.dump(time_results, f, indent=4, ensure_ascii=False)
        print(f"⏱ Tempos salvos em: {time_filename}")

        # Salvar modelos usados
        with open(models_filename, "w", encoding="utf-8") as f:
            json.dump(models_used, f, indent=4, ensure_ascii=False)
        print(f"📄 Modelos usados salvos em: {models_filename}")

        print(f"⏳ Tempo total para {config_name}: {time.time() - start_time:.2f} segundos.")

    print("\n✅ Geração de respostas, tempos e modelos concluída para todas as configurações!")

if __name__ == "__main__":
    start_time = time.time()
    gerar_respostas()
    print(f"\n⏳ Tempo total de execução: {time.time() - start_time:.2f} segundos.")