import os
import time
import traceback
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
INDEX_FAISS_DIR = "data/embeddings/"

def stage_1_scraper():
    """Scrapes contest notices (editais) and saves them as CSV."""
    print("\n🔍 [1/6] Collecting contest notices...")
    editais_info = fetch_edital_links(BASE_URL, "data/raw/")
    df_editais = pd.DataFrame(editais_info)
    df_editais.to_csv(CSV_EDITAIS, index=False)
    print(f"✅ Notices saved at: {CSV_EDITAIS}")

def stage_2_extraction():
    """Extracts text from PDFs and splits them into chunks."""
    print("\n📄 [2/6] Extracting text from PDFs...")

    files = [
        ("data/raw/ibge.txt", "IBGE"),
        ("data/extracted_pedro/cnen.txt", "CNEN"),
        ("data/extracted_pedro/aeronautica.txt", "Aeronáutica"),
        ("data/extracted_pedro/AEB-agencia-espacial-brasileira.txt", "AEB"),
        ("data/extracted_pedro/ibama.txt", "IBAMA"),
        ("data/extracted_pedro/edital_funai.txt", "FUNAI"),
        ("data/extracted_pedro/edital_trf.txt", "TRF"),
        ("data/extracted_pedro/edital_marinha_50pag.txt", "Marinha"),
    ]

    all_chunks = []
    contests = []

    for file, contest in files:
        chunks = chunking_texto(file)
        all_chunks.extend(chunks)
        contests.extend([contest] * len(chunks))  # Associate each chunk with the correct contest

    df_results = pd.DataFrame({"Chunk": all_chunks, "Concurso": contests})
    
    print("🔍 Verifying columns before saving:", df_results.columns)
    
    df_results.to_csv(CSV_CHUNKS, index=False)
    print(f"✅ Extracted texts saved at: {CSV_CHUNKS}")

def file_exists(path):
    """Checks if a file exists before using it."""
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return False
    return True

def stage_3_embeddings():
    """Generates embeddings and creates FAISS indices per contest."""
    print("\n🧠 [3/6] Creating embeddings and FAISS indices...")

    if not file_exists(CSV_CHUNKS):
        print("⚠️ Skipping embeddings stage as the text chunks were not extracted.")
        return

    df_chunks = pd.read_csv(CSV_CHUNKS).dropna().reset_index(drop=True)

    store = FAISSVectorStore()
    store.create_indices()
    store.print_faiss_status()

    print(f"✅ FAISS indices saved at: {INDEX_FAISS_DIR}")

def stage_4_test_rag():
    """Executa testes no sistema RAG usando FAISS corretamente."""
    print("\n🤖 [4/6] Testando RAG...")

    faiss_store = FAISSVectorStore()
    faiss_store.load_indices()

    rag = RAGPipeline()

    questions = [
        "Quantas vagas estão disponíveis no concurso da FUNAI?",
        "Me fale sobre o calendário da FUNAI",
        "Qual a carga horária para os cargos do IBAMA?",
        "Quais os requisitos para o concurso da Marinha?",
        "Onde os candidatos devem se inscrever no concurso da Aeronáutica?",
        "Qual o salário do concurso do IBGE?"
    ]

    responses = {}

    for question in questions:
        try:
            print(f"\n🔹 Pergunta: {question}")
            response = rag.generate_answer(question)
            print(f"💬 Resposta: {response}")
            responses[question] = str(response)

        except Exception as e:
            print(f"❌ Erro ao gerar resposta para '{question}': {e}")
            traceback.print_exc()

    save_results(responses)



def execute_script(script_path):
    """Runs a Python script and displays real-time output."""
    if not os.path.exists(script_path):
        print(f"⚠️ File not found: {script_path}")
        return

    print(f"▶️ Running: {script_path} ...")

    try:
        process = subprocess.Popen(["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in iter(process.stdout.readline, ''):
            print(line, end="")  # Display script output in real-time
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            print(f"⚠️ Error executing {script_path}. Return code: {return_code}")
    except Exception as e:
        print(f"❌ Error running {script_path}: {e}")

def stage_5_experiments():
    """Runs experiments on embeddings, chunking, and LLMs."""
    print("\n📊 [5/6] Running experiments...\n")

    experiment_scripts = [
        "experiments/embeddings_experiment.py",
        "experiments/chunking_experiment.py",
        "experiments/llm_experiment.py"
    ]

    for script in experiment_scripts:
        execute_script(script)

    print("\n✅ Experiments completed!")

def stage_6_metrics():
    """Evaluates chatbot performance and saves metrics."""
    print("\n📈 [6/6] Evaluating chatbot performance...\n")
    
    start_time = time.time()
    
    try:
        avaliar_sistema()
        print("✅ Evaluation completed! Check results at 'reports/metricas.csv'")
    except Exception as e:
        print(f"❌ Error evaluating metrics: {e}")
    
    total_time = time.time() - start_time
    print(f"\n⏳ Total evaluation time: {total_time:.2f} seconds.")

def start_ui():
    """Starts the Streamlit web interface."""
    print("\n🚀 Starting web interface...")
    os.system("streamlit run ui/app.py")

if __name__ == "__main__":
    start_time = time.time()

    # Run the pipeline steps
    
    # stage_1_scraper()
    #stage_2_extraction()
    #stage_3_embeddings()
    stage_4_test_rag()
    #stage_5_experiments()
    # stage_6_metrics()

    total_time = time.time() - start_time
    print(f"\n⏳ Total execution time: {total_time:.2f} seconds.")

    if "--ui" in sys.argv:
        start_ui()
