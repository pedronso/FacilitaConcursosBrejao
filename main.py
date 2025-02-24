import os
import time
import pandas as pd
from generate.generate_chunks import processar_chunks
from generate.generate_faiss import criar_faiss_index, criar_faiss_index_unit
from generate.generate_folders import criar_pastas
from generate.generate_metadata import atualizar_metadados
from generate.generate_responses import gerar_respostas
from generate.verify_embedding_models import verificar_modelos_de_embeddings
from pipelines.scraper import fetch_edital_links
from pipelines.extractor import processar_downloads_e_extração, chunking_texto
from models.embeddings_model import EmbeddingModel
from vectorstore.faiss_store import FAISSVectorStore
from pipelines.rag import RAGPipeline
from reports.metrics import avaliar_sistema
from experiments.results_saver import save_results
from experiments.result_verifier import METRICS_DIR, ResultVerifier
import subprocess
import sys
import json
import tests_vars
from convert_json_utf8 import process_all_json_files



BASE_URL = "https://www.pciconcursos.com.br/concursos/nacional/"
CSV_EDITAIS = "data/processed/editais_concursos.csv"
CSV_CHUNKS = "data/processed/results_extraction_chunks_updated.csv"
INDEX_FAISS = "data/embeddings/faiss_index_1"
#CSV_CHUNKS = "data/processed/results_extraction_chunks.csv"
# INDEX_FAISS = "data/embeddings/faiss_index"

def etapa_1_scraper():
    """Baixa os PDFs dos editais e salva no CSV."""
    print("\n🔍 [1/6] Coletando editais...")
    editais_info = fetch_edital_links(BASE_URL, "data/raw/")
    df_editais = pd.DataFrame(editais_info)
    df_editais.to_csv(CSV_EDITAIS, index=False)
    print(f"✅ Editais salvos em: {CSV_EDITAIS}")

def etapa_2_extracao():
    """Extrai texto dos PDFs e divide em chunks."""
    print("\n📄 [2/6] Extraindo textos dos PDFs...")
    #resultados = processar_downloads_e_extração(CSV_EDITAIS, "data/raw/")
    #resultados = chunking_texto("data/raw/text.txt")
    files = [
        "data/raw/ibge.txt",
    ]
    files = [
        "data/extracted_pedro/aeb.txt",
        "data/extracted_pedro/aeronautica.txt",
        "data/extracted_pedro/cceb.txt",
        "data/extracted_pedro/cnen.txt",
        "data/extracted_pedro/funai.txt",
        "data/extracted_pedro/ibama.txt",
        "data/extracted_pedro/ibge.txt",
        "data/extracted_pedro/marinha.txt",
        "data/extracted_pedro/mpu.txt",
        "data/extracted_pedro/trf.txt",
    ]

    all_chunks = []
    for file in files:
        chunks = chunking_texto(file)
        index = len(all_chunks) + 1
        print(index)
        tests_vars.idx.append(index)
        all_chunks.extend(chunks)  # Add chunks from each file to the list

    print("chunks!!: \n")
    print(tests_vars.idx)
    # print(len(chunks))
    print("\n\n")
    # Create DataFrame with one chunk per row
    df_resultados = pd.DataFrame({"Chunk": all_chunks})
    df_resultados.to_csv(CSV_CHUNKS, index=False)

    #df_resultados = pd.DataFrame(resultados)
    # df_resultados.to_csv(CSV_CHUNKS, index=False)
    # df_resultados = pd.DataFrame({"Chunk": [texto_unico]})
    print(f"✅ Textos extraídos e salvos em: {CSV_CHUNKS}")

def verificar_existencia_arquivo(caminho):
    """Verifica se o arquivo existe antes de tentar utilizá-lo."""
    if not os.path.exists(caminho):
        print(f"❌ Arquivo não encontrado: {caminho}")
        return False
    return True

def etapa_3_embeddings():
    """Gera embeddings e cria o banco FAISS."""
    print("\n🧠 [3/6] Criando embeddings e index FAISS...")

    if not verificar_existencia_arquivo(CSV_CHUNKS):
        print("⚠️ Pulando etapa de embeddings pois os chunks não foram extraídos.")
        return


    """
    if "Chunks" not in df_chunks.columns:
        print("⚠️ A coluna 'Chunks' não foi encontrada no CSV. Verifique os dados processados.")
        return
    """

    # Verifica se a coluna 'Chunks' é string e converte para lista se necessário
    """
    if isinstance(df_chunks["Chunks"].iloc[0], str):
        try:
            df_chunks["Chunks"] = df_chunks["Chunks"].apply(eval)
        except:
            print("⚠️ Erro ao converter 'Chunks' para lista.")
            return
    """

    #print(df_original.head())  # Ver o início do DataFrame
    #print(df_original.shape)   # Ver o número de linhas e colunas
    #print(df_original.columns) # Ver os nomes das colunas

    #df_chunks = df_original.melt(var_name="Chunk_Index", value_name="Chunk").dropna().reset_index(drop=True)
    df_original = pd.read_csv(CSV_CHUNKS)
    df_chunks = df_original.dropna().reset_index(drop=True)

    chunks = df_original['Chunk'].dropna().tolist()

    print(df_chunks.columns)
    print(df_chunks.head())
    
    store = FAISSVectorStore()
    #texts = [" ".join(map(str, row.dropna())) for _, row in df_chunks.iterrows()]
    #store.create_index([" ".join(chunk) if isinstance(chunk, list) else chunk for chunk in df_chunks["Chunks"]])
    store.create_index(chunks)


    print(f"✅ FAISS index salvo em: {INDEX_FAISS}")

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

CONFIGS_DIR = "data/processed/configs/"
RESULTS_DIR = "data/processed/results/"

def etapa_4_testar_rag():
    """Executa testes no chatbot RAG para todas as configurações geradas."""
    if not os.path.exists(CONFIGS_DIR):
        print(f"❌ Diretório de configurações não encontrado: {CONFIGS_DIR}")
        return

    # Criar diretório de resultados se não existir
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Iterar sobre cada configuração disponível
    for config_name in os.listdir(CONFIGS_DIR):
        config_path = os.path.join(CONFIGS_DIR, config_name, "config.json")
        chunks_path = os.path.join(CONFIGS_DIR, config_name, "chunks.csv")
        faiss_path = os.path.join(CONFIGS_DIR, config_name, "faiss_index")

        if not os.path.exists(config_path) or not os.path.exists(chunks_path) or not os.path.exists(faiss_path):
            print(f"⚠️ Ignorando configuração incompleta: {config_name}")
            continue

        print(f"\n🔍 Testando configuração: {config_name}")

        # Criar pipeline RAG com caminhos específicos para essa configuração
        rag = RAGPipeline(
            faiss_index_path=faiss_path,
            chunks_csv_path=chunks_path
        )

        perguntas_respostas_dict = {}

        for pergunta in perguntas:
            try:
                print(f"\n🔹 Pergunta: {pergunta}")
                resposta = rag.generate_answer(pergunta)
                print(f"💬 Resposta: {resposta}")
                perguntas_respostas_dict[pergunta] = str(resposta)

            except Exception as e:
                print(f"❌ Erro ao gerar resposta para '{pergunta}': {e}")

        # Salvar resultados específicos dessa configuração
        results_filename = f"{RESULTS_DIR}/{config_name}_results.json"
        with open(results_filename, "w", encoding="utf-8") as f:
            json.dump(perguntas_respostas_dict, f, indent=4)

        print(f"📁 Resultados salvos em: {results_filename}")

    print("\n✅ Testes concluídos para todas as configurações!")

    
def etapa_4_testar_rag_old():
    """Executa um teste no chatbot RAG."""
    print("\n🤖 [4/6] Testando consultas ao sistema RAG...")
    
    rag = RAGPipeline()
    perguntas_respostas_dict = {}

    for pergunta in perguntas:
        try:
            print(f"\n🔹 Pergunta: {pergunta}")
            resposta = rag.generate_answer(pergunta)
            print(f"💬 Resposta: {resposta}")
            perguntas_respostas_dict[pergunta] = str(resposta)

        except Exception as e:
            print(f"❌ Erro ao gerar resposta para '{pergunta}': {e}")
    
    save_results(perguntas_respostas_dict)

def run_script(script_path):
    """Executa um script Python e exibe a saída em tempo real."""
    if not os.path.exists(script_path):
        print(f"⚠️ Arquivo não encontrado: {script_path}")
        return

    print(f"▶️ Executando: {script_path} ...")

    try:
        process = subprocess.Popen(["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in iter(process.stdout.readline, ''):
            print(line, end="")  # Exibe a saída do script em tempo real
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            print(f"⚠️ Erro ao executar {script_path}. Código de retorno: {return_code}")
    except Exception as e:
        print(f"❌ Erro ao rodar {script_path}: {e}")

def etapa_5_experimentos():
    """Executa os experimentos de embeddings, chunking e LLMs."""
    print("\n📊 [5/6] Executando experimentos...\n")

    scripts_experimentos = [
        "experiments/embeddings_experiment.py",
        "experiments/chunking_experiment.py",
        "experiments/llm_experiment.py"
    ]

    for script in scripts_experimentos:
        run_script(script)

    print("\n✅ Experimentos concluídos!")

def etapa_6_metricas():
    """Avalia desempenho do sistema e salva métricas."""
    print("\n📈 [6/6] Avaliando desempenho do chatbot...\n")
    
    start_time = time.time()
    
    try:
        avaliar_sistema()
        print("✅ Avaliação concluída! Veja os resultados em 'reports/metricas.csv'")
    except Exception as e:
        print(f"❌ Erro ao avaliar métricas: {e}")
    
    total_time = time.time() - start_time
    print(f"\n⏳ Tempo total da avaliação: {total_time:.2f} segundos.")


def etapa_4_1_avaliar_rag():
    reviewer = ResultVerifier()
    reviewer.review()



def iniciar_interface():
    """Executa a interface Streamlit."""
    print("\n🚀 Iniciando interface web...")
    os.system("streamlit run ui/app.py")


def executar_pipeline_completa(filename=None):
    """Executa todas as etapas na ordem correta."""
    start_time = time.time()

    #flow antigo
    #etapa_1_scraper()
    #etapa_2_extracao()
    #etapa_3_embeddings()
    #etapa_4_testar_rag()
    #etapa_4_1_avaliar_rag()
    #etapa_5_experimentos()
    #etapa_6_metricas()



    #flow novo
    #print("🚀 Criando estrutura de diretórios...")
    #criar_pastas()

    #print("🔹 Gerando chunks...")
    #processar_chunks()

    #print("🧠 Criando índices FAISS...")
    #criar_faiss_index()

    #print("\n🚀 Gerando respostas para todas as configurações...")
    #gerar_respostas()  # Chama diretamente a função de geração de respostas

    # Converter todos os JSONs para UTF-8 antes de processá-los
    #process_all_json_files()
    
    #print("\n📊 Avaliando as respostas geradas...")
    #verifier = ResultVerifier()
    #verifier.review_new_structure()  # Avalia respostas na nova estrutura

    #print("\n📊 Reavaliando as médias geradas...")
    #if filename:
    #    metric_filepath = os.path.join(METRICS_DIR, filename)
    #    if os.path.exists(metric_filepath):
    #        verifier.corrigir_media_arquivo(metric_filepath)
    #    else:
    #        print(f"⚠️ Arquivo {filename} não encontrado. Nenhuma correção aplicada.")
    #else:
    #    verifier.corrigir_todas_as_medias()

    print("\n✅ Processos finalizados!")

    total_time = time.time() - start_time
    print(f"\n⏳ Tempo total de execução: {total_time:.2f} segundos.")

    if "--ui" in sys.argv:
        iniciar_interface()

if __name__ == "__main__":
    executar_pipeline_completa()