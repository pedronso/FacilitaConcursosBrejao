import os
import time
import pandas as pd
from pipelines.scraper import fetch_edital_links
from pipelines.extractor import processar_downloads_e_extração
from models.embeddings_model import EmbeddingModel
from vectorstore.faiss_store import FAISSVectorStore
from pipelines.rag import RAGPipeline
from reports.metrics import avaliar_sistema
import subprocess
import sys
import json
import tests_vars


BASE_URL = "https://www.pciconcursos.com.br/concursos/nacional/"
CSV_EDITAIS = "data/processed/editais_concursos.csv"
CSV_CHUNKS = "data/processed/results_extraction_chunks.csv"
INDEX_FAISS = "data/embeddings/faiss_index"

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
    resultados = processar_downloads_e_extração(CSV_EDITAIS, "data/raw/")
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(CSV_CHUNKS, index=False)
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

    df_original = pd.read_csv(CSV_CHUNKS)
    df_chunks = df_original.melt(var_name="Chunk_Index", value_name="Chunk").dropna().reset_index(drop=True)

    print(df_chunks.columns)
    print(df_chunks.head())
    
    store = FAISSVectorStore()
    #texts = [" ".join(map(str, row.dropna())) for _, row in df_chunks.iterrows()]
    #store.create_index([" ".join(chunk) if isinstance(chunk, list) else chunk for chunk in df_chunks["Chunks"]])
    store.create_index(df_chunks['Chunk'].to_list())


    print(f"✅ FAISS index salvo em: {INDEX_FAISS}")

def etapa_4_testar_rag( ):
    """Executa um teste no chatbot RAG."""
    print("\n🤖 [4/6] Testando consultas ao sistema RAG...")
    
    rag = RAGPipeline()
    """
    queries_teste = [
        "Quais são os prazos de inscrição do concurso da marinha?",
        "Quais são os concursos para engenheiros?",
        "Tem algum concurso para nível médio?",
        "Quantas vagas estão abertas para o IBAMA?",
        "O MPU está com concursos abertos?"
    ]
    """
    concursos = [
        "da aeronáutica",
        "da PPSA",
        "do Ibama",
        "da Marinha",
        "do MPU",
        "da AEB",
        "da CNEN",
        "da EBSERH",
        "da FUNAI",
        "do IBGE",
        "do ICMBio",
        "do TRF"
    ]
    queries_teste =[
        "Qual a data da prova para o concurso",
        "Qual é a data da inscrição para o concurso",
        "Qual é o salário para o concurso",
        "Qual é a quantidade de vagas para o concurso",
        "Quais são os assuntos da prova do concurso",
        "Qual é a carga horária para os cargos do concurso"
    ]

    perguntas = []
    perguntas_repostas_dict = {}

    for concurso in concursos:
        for querie in queries_teste:
            pergunta = f'{querie} {concurso}'

            perguntas.append(pergunta)

    #print(perguntas)

    for pergunta in perguntas:
        try:
            print(f"\n🔹 Pergunta: {pergunta}")
            #resposta_local = rag.generate_full_answer(query)
            resposta = rag.generate_answer(pergunta)
            print(f"💬 Resposta: {resposta}")
            #print(f"💬 Resposta: {resposta_local}")
            perguntas_repostas_dict[pergunta] = str(resposta)
            json_string = json.dumps(perguntas_repostas_dict, indent=4, ensure_ascii=False)
            save_local = f'{tests_vars.dict_models['ai_model']}-{tests_vars.dict_models['embedding_model'].replace('/', '-')}-{tests_vars.dict_models['chunk_size']}'
            with open(f'data/responses/{save_local}.json', 'w', encoding='utf-8') as arquivo:
                arquivo.write(json_string)
            #sleep(1)
        except Exception as e:
            print(f"❌ Erro ao gerar resposta para '{pergunta}': {e}")

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


def etapa_7_avaliar_respostas():

    pass


def iniciar_interface():
    """Executa a interface Streamlit."""
    print("\n🚀 Iniciando interface web...")
    os.system("streamlit run ui/app.py")

if __name__ == "__main__":
    start_time = time.time()

    #etapa_1_scraper()
    #etapa_2_extracao()
    #etapa_3_embeddings()
    #etapa_4_testar_rag()
    #etapa_5_experimentos()
    #etapa_6_metricas()
    etapa_7_avaliar_respostas()

    total_time = time.time() - start_time
    print(f"\n⏳ Tempo total de execução: {total_time:.2f} segundos.")

    if "--ui" in sys.argv:
        iniciar_interface()
