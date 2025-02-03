import requests
import os
import pandas as pd
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
import re

def download_pdf(pdf_url, pasta_destino):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        pdf_nome = pdf_url.split('/')[-1]
        pdf_path = os.path.join(pasta_destino, pdf_nome)

        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        print(f"PDF {pdf_nome} baixado com sucesso!")
        
        return pdf_path

    except Exception as e:
        print(f"Erro ao baixar o PDF de {pdf_url}: {e}")
        return None

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

def dividir_em_chunks(texto, tamanho_maximo=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=tamanho_maximo, chunk_overlap=200)
    chunks = text_splitter.split_text(texto)
    return chunks

def processar_downloads_e_extração(csv_path, pasta_destino):
    df = pd.read_csv(csv_path)

    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    resultados = []

    for _, row in df.iterrows():
        if isinstance(row['PDFs'], str):
            try:
                pdf_links = eval(row['PDFs'])
            except:
                pdf_links = []

            textos_extraidos = []

            for pdf_link in pdf_links:
                pdf_path = download_pdf(pdf_link, pasta_destino)
                
                if pdf_path:
                    texto_pdf = extrair_texto_pdf(pdf_path)
                    texto_limpo = limpar_texto(texto_pdf)
                    textos_extraidos.append(texto_limpo)

            texto_completo = " ".join(textos_extraidos)
            
            chunks = dividir_em_chunks(texto_completo)

            resultados.append({
                'Título': row['Título'],
                'Detalhes': row['Detalhes'],
                'PDFs': row['PDFs'],
                'Texto Extraído': texto_completo,
                'Número de Chunks': len(chunks),
                'Chunks': chunks,
            })

    return resultados

csv_path = 'editais_concursos.csv'
pasta_destino = 'pdfs'

resultados = processar_downloads_e_extração(csv_path, pasta_destino)

for resultado in resultados:
    print(f"\nTítulo: {resultado['Título']}")
    print(f"Detalhes: {resultado['Detalhes']}")
    print(f"Número de Chunks: {resultado['Número de Chunks']}")
    print(f"Primeiro Chunk: {resultado['Chunks'][0][:300]}\n")

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("resultados_extracao.csv", index=False, encoding='utf-8')
