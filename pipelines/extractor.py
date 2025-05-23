import os
import pandas as pd
from pdfminer.high_level import extract_text
from langchain.text_splitter import TokenTextSplitter
import re
import nltk
from nltk.corpus import stopwords
from tests_vars import dict_models

nltk.download('stopwords')

STOPWORDS = set(stopwords.words('portuguese'))

def extrair_texto_pdf(pdf_path):
    """Extrai texto de um PDF."""
    try:
        texto = extract_text(pdf_path)
        return texto
    except Exception as e:
        print(f"❌ Erro ao extrair texto do PDF {pdf_path}: {e}")
        return ""
    
def limpar_texto(texto):
    """Remove espaços e caracteres desnecessários do texto extraído."""
    texto_limpo = re.sub(r'\s+', ' ', texto).strip()
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

def remover_stopwords(texto):
    """Remove stopwords do texto extraído."""
    palavras = texto.split()
    texto_sem_stopwords = " ".join([palavra for palavra in palavras if palavra.lower() not in STOPWORDS])
    return texto_sem_stopwords

def aplicar_stemming(texto):
    """Aplica stemming às palavras do texto"""
    palavras = texto.split()
    texto_stemmed = " ".join([STEMMER.stem(palavra) for palavra in palavras])
    return texto_stemmed

def chunking_texto(file_path, labeled, normalized, remove_stopwords, chunk_size, chunk_overlap):
    """Processa o arquivo e cria chunks de acordo com os parâmetros."""
    
    with open(file_path, "r", encoding="utf-8") as arquivo:
        texto_completo = arquivo.read()
        if normalized:
            texto_completo = texto_completo.lower()
        if remove_stopwords:
            texto_completo = remover_stopwords(texto_completo)

        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(texto_completo)

    return chunks

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
    
