import os
import pandas as pd
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
import re
from tests_vars import dict_models

def extrair_texto_pdf(pdf_path):
    try:
        texto = extract_text(pdf_path)
        return texto
    except Exception as e:
        print(f"Erro ao extrair texto do PDF {pdf_path}: {e}")
        return ""

def limpar_texto(texto):
    texto_limpo = re.sub(r'\s+', ' ', texto)
    texto_limpo = texto_limpo.strip()
    return texto_limpo

def dividir_em_chunks(texto, rotulo, com_rotulo=dict_models['labeled'], tamanho_maximo=dict_models['chunk_size'], chunk_overlap=dict_models['chunk_overlap']):
    # tamanho_rotulo = len(f'[{rotulo}] ') if com_rotulo else 0
    
    # tamanho_ajustado = tamanho_maximo-tamanho_rotulo

    text_splitter = TokenTextSplitter(chunk_size=tamanho_maximo, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(texto)

    chunks_tratados = [f'[{rotulo}] {chunk}' for chunk in chunks] if com_rotulo else chunks
    return chunks_tratados

def processar_downloads_e_extração(csv_path, pasta_destino):
    df = pd.read_csv(csv_path)

    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    resultados = []

    for _, row in df.iterrows():
        if isinstance(row['PDFs'], str):
            try:
                pdf_paths = eval(row['PDFs'])
            except:
                pdf_paths = []

            textos_extraidos = []
            for pdf_path in pdf_paths:
                print(pdf_path)
                full_pdf_path = pdf_path[3:]

                if full_pdf_path and os.path.exists(full_pdf_path):
                    print("existe o path")
                    texto_pdf = extrair_texto_pdf(full_pdf_path)
                    texto_limpo = limpar_texto(texto_pdf)
                    textos_extraidos.append(texto_limpo)
                else:
                    print(f"nao existe o path{full_pdf_path}: {pdf_path}")

            texto_completo = " ".join(textos_extraidos)
            
            chunks = dividir_em_chunks(texto_completo, row['Título'])
            print(chunks)

            """
            resultados.append({
                'Título': row['Título'],
                'Detalhes': row['Detalhes'],
                'PDFs': row['PDFs'],
                #'Texto Extraído': texto_completo,
                'Número de Chunks': len(chunks),
                'Chunks': chunks,
            })
            """
            resultados.append(chunks)

    return resultados

def chunking_texto(file_path):
    #resultados = []
    with open(file_path, "r", encoding="utf-8") as arquivo:
        texto_completo = arquivo.read()
        chunks = dividir_em_chunks(texto_completo, file_path.split("/")[-1].replace(".txt", ""))
        #print(chunks)
        #resultados.append(chunks)
    
    return chunks

    return resultados

if __name__ == "__main__":
    csv_path = 'data/processed/editais_concursos.csv'
    pasta_destino = 'data/raw/'

    # resultados = processar_downloads_e_extração(csv_path, pasta_destino)
    files = [
        "data/extracted_pedro/aeronautica.txt",
        "data/extracted_pedro/ibama.txt",
        "data/extracted_pedro/agencia.txt",
        "data/extracted_pedro/CCEB- Censo Cidades Estudantil Brasil.txt",
        "data/extracted_pedro/cnen.txt",
        "data/raw/text.txt"
    ]
    resultados = [" ".join(chunking_texto(file)) for file in files]

    for resultado in resultados:
        continue
        print(f'{resultado}')
        print(f"\nTítulo: {resultado['Título']}")
        print(f"Detalhes: {resultado['Detalhes']}")
        print(f"Número de Chunks: {resultado['Número de Chunks']}")
        print(f"Primeiro Chunk: {resultado['Chunks'][0][:300]}\n")


    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("data/processed/results_extraction_chunks.csv", index=False, encoding='utf-8')
    
