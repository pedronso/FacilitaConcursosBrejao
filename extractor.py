import os
import pandas as pd
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

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

def dividir_em_chunks(texto, tamanho_maximo=512):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=tamanho_maximo, chunk_overlap=200)
    chunks = text_splitter.split_text(texto)
    return chunks

def processar_downloads_e_extração(csv_path, pasta_destino):
    df = pd.read_csv(csv_path)

    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    resultados = []

    doc_count = 1
    for _, row in df.iterrows():

        if isinstance(row['PDFs'], str) and doc_count > 0:
            try:
                pdf_paths = eval(row['PDFs'])
            except:
                pdf_paths = []

            textos_extraidos = []
            

            for pdf_path in pdf_paths:
                print(pdf_path, _, row)
                if pdf_path and os.path.exists(pdf_path):
                    texto_pdf = extrair_texto_pdf(pdf_path)
                    texto_limpo = limpar_texto(texto_pdf)
                    textos_extraidos.append(texto_limpo)
                
            texto_completo = " ".join(textos_extraidos)
            
            chunks = dividir_em_chunks(texto_completo)

            """
            resultados.append({
                'Título': row['Título'],
                'Detalhes': row['Detalhes'],
                'PDFs': row['PDFs'],
                'Número de Chunks': len(chunks),
                'Chunks': chunks,
            })
            """
            doc_count -= 1
            resultados.extend(chunks)

    return pd.DataFrame(resultados, columns=["Chunk"])

csv_path = 'editais_concursos.csv'
pasta_destino = 'pdfs'

df_resultados = processar_downloads_e_extração(csv_path, pasta_destino)
df_resultados.to_csv("resultados_extracao.csv", index=False, encoding='utf-8')
print(df_resultados.head())