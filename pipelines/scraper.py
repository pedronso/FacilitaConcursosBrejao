import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

BASE_URL = "https://www.pciconcursos.com.br/concursos/nacional/"
PASTA_DESTINO = "data/raw/"

def download_pdf(pdf_url, pasta_destino):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        pdf_nome = pdf_url.split('/')[-1]
        pdf_path = os.path.join(pasta_destino, pdf_nome)

        if not os.path.exists(pasta_destino):
            os.makedirs(pasta_destino)

        # evitar repetição
        if not os.path.exists(pdf_path):
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            print(f"PDF {pdf_nome} baixado com sucesso!")
        else:
            print(f"PDF {pdf_nome} já existe, não será baixado novamente.")
        
        return pdf_path

    except Exception as e:
        print(f"Erro ao baixar o PDF de {pdf_url}: {e}")
        return None

def fetch_edital_links(base_url, pasta_destino):
    """Obtém os links das páginas de detalhes dos editais e baixa os PDFs.""" 
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    editais = []

    edital_divs = soup.find_all('div', attrs={'data-url': True})

    for edital_div in edital_divs:
        titulo = edital_div.find('div', class_='ca').find('a').get_text(strip=True)
        descricao = edital_div.find('div', class_='ca').find('a')['title']
        link_pagina_edital = edital_div['data-url']

        pdfs = fetch_pdf_links(link_pagina_edital)

        editais.append({
            'Título': titulo,
            'Detalhes': descricao,
            'Link Página Edital': link_pagina_edital,
            'PDFs': pdfs,
        })

    return editais

def fetch_pdf_links(edital_url):
    """Acessa a página do edital e retorna uma lista de todos os PDFs encontrados, fazendo download deles."""
    response = requests.get(edital_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    pdf_links = [a['href'] for a in soup.find_all('a', href=lambda href: href and href.endswith('.pdf'))]

    pdf_paths = [download_pdf(link, PASTA_DESTINO) for link in pdf_links]

    return pdf_paths if pdf_paths else ["Nenhum PDF encontrado"]

pasta_destino = 'pdfs'
editais_info = fetch_edital_links(BASE_URL, pasta_destino)

df_editais = pd.DataFrame(editais_info)
df_editais.to_csv("editais_concursos.csv", index=False, encoding='utf-8')

print(df_editais.head())
