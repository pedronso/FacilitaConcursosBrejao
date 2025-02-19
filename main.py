import os
import time
import pandas as pd
from pipelines.scraper import fetch_edital_links
from pipelines.extractor import processar_downloads_e_extração, chunking_texto
from models.embeddings_model import EmbeddingModel
from vectorstore.faiss_store import FAISSVectorStore
from pipelines.rag import RAGPipeline
from reports.metrics import avaliar_sistema
from experiments.results_saver import save_results
import subprocess
import sys
import json
import tests_vars


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
        "data/extracted_pedro/ibge.txt",
        "data/extracted_pedro/cnen.txt",
        "data/extracted_pedro/cceb.txt",
        "data/extracted_pedro/aeronautica.txt",
        "data/extracted_pedro/aeb.txt",
        "data/extracted_pedro/ibama.txt",
        "data/extracted_pedro/funai.txt",
        "data/extracted_pedro/trf.txt",
        "data/extracted_pedro/marinha.txt"
    ]

    all_chunks = []
    for file in files:
        chunks = chunking_texto(file)
        all_chunks.extend(chunks)  # Add chunks from each file to the list

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
    """
    concursos = [
        "do IBGE",
    ]
    queries_teste =[
        "ANEXO I - Quadro de Vagas e Postos de Inscrição do",
        "Como funciona a classificação e titulação para o concurso",
        "Qual é a data da inscrição para o concurso",
        "Qual é o salário para o concurso",
        "Qual é a quantidade de vagas para o concurso",
        "Qual é a carga horária para os cargos do concurso"
    ]
    """

    perguntas_respostas_dict = {}
    """
    for concurso in concursos:
        for querie in queries_teste:
            pergunta = f'{querie} {concurso}'

            perguntas.append(pergunta)
    """
    #print(perguntas)
    perguntas = [
                 "Qual é o cronograma completo do processo seletivo da fundação nacional do índio, desde as inscrições até a divulgação do resultado final?",
                 "Qual é o cronograma completo do processo seletivo do instituto brasileiro do meio ambiente, desde as inscrições até a divulgação do resultado final?",
                 "Qual é o cronograma completo do processo seletivo do trf, desde as inscrições até a divulgação do resultado final?",
                 "Qual é o cronograma completo do processo seletivo do instituto brasileiro de geografia e estatística, desde as inscrições até a divulgação do resultado final?",
                 "Qual é o cronograma completo do processo seletivo do concurso da energia nuclear, desde as inscrições até a divulgação do resultado final?",
                 "Qual é o cronograma completo do processo seletivo do censo cidades estudantil brasil, desde as inscrições até a divulgação do resultado final?",
                 "Qual é o cronograma completo do processo seletivo do fab, desde as inscrições até a divulgação do resultado final?",
                 "Qual é o cronograma completo do processo seletivo do agência espacial brasileira, desde as inscrições até a divulgação do resultado final?",
                #"No concurso do ibama Quantas vagas estão sendo oferecidas no total e como elas estão distribuídas entre os municípios?",
                # "Qual é a remuneração mensal para as funções de Agente de Pesquisas e Mapeamento e Supervisor de Coleta e Qualidade?",
                # "Sobre o concurso do ibama Como e onde as inscrições devem ser realizadas?",
                # "Quais documentos são necessários para a inscrição e quais devem ser apresentados no momento da contratação do ibama?",
                # "Onde os candidatos podem obter informações adicionais sobre o processo seletivo do ibama?"
                ]
    
    for pergunta in perguntas:
        try:
            print(f"\n🔹 Pergunta: {pergunta}")
            #resposta_local = rag.generate_full_answer(query)
            resposta = rag.generate_answer(pergunta)
            print(f"💬 Resposta: {resposta}")
            #print(f"💬 Resposta: {resposta_local}")
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
    etapa_4_testar_rag()
    #etapa_5_experimentos()
    #etapa_6_metricas()
    #etapa_7_avaliar_respostas()

    total_time = time.time() - start_time
    print(f"\n⏳ Tempo total de execução: {total_time:.2f} segundos.")

    if "--ui" in sys.argv:
        iniciar_interface()
