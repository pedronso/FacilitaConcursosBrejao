import os
import time
import pandas as pd
from pipelines.scraper import fetch_edital_links
from pipelines.extractor import processar_downloads_e_extração
from models.embeddings_model import EmbeddingModel
from vectorstore.faiss_store import FAISSVectorStore
from pipelines.rag import RAGPipeline
from reports.metrics import avaliar_sistema
import streamlit.cli as stcli
import sys
import subprocess



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

def etapa_3_embeddings():
    """Gera embeddings e cria o banco FAISS."""
    print("\n🧠 [3/6] Criando embeddings e index FAISS...")
    df_chunks = pd.read_csv(CSV_CHUNKS)
    store = FAISSVectorStore()
    store.create_index(df_chunks["Chunks"].tolist())
    print(f"✅ FAISS index salvo em: {INDEX_FAISS}")

def etapa_4_testar_rag():
    """Executa um teste no chatbot RAG."""
    print("\n🤖 [4/6] Testando consultas ao sistema RAG...")
    rag = RAGPipeline()
    queries_teste = [
        "Quais são os concursos para engenheiros?",
        "Quais os prazos de inscrição?",
        "Tem algum concurso para nível médio?"
    ]
    
    for query in queries_teste:
        resposta = rag.generate_answer(query)
        print(f"\n🔹 Pergunta: {query}")
        print(f"💬 Resposta: {resposta}")

def run_script(script_path):
    """Executa um script Python e exibe a saída em tempo real."""
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
        if os.path.exists(script):
            print(f"▶️ Executando: {script} ...")
            run_script(script)
        else:
            print(f"⚠️ Arquivo não encontrado: {script}")

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

def iniciar_interface():
    """Executa a interface Streamlit."""
    print("\n🚀 Iniciando interface web...")
    os.system("streamlit run ui/app.py")

if __name__ == "__main__":
    start_time = time.time()

    #etapa_1_scraper()
    #etapa_2_extracao()
    etapa_3_embeddings()
    etapa_4_testar_rag()
    etapa_5_experimentos()
    etapa_6_metricas()

    total_time = time.time() - start_time
    print(f"\n⏳ Tempo total de execução: {total_time:.2f} segundos.")

    if "--ui" in sys.argv:
        iniciar_interface()
