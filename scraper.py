import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL da página principal
BASE_URL = "https://www.pciconcursos.com.br/concursos/nacional/"

def fetch_edital_links(base_url):
    """Obtém os links das páginas de detalhes dos editais."""
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    editais = []

    edital_divs = soup.find_all('div', attrs={'data-url': True})

    for edital_div in edital_divs:
        titulo = edital_div.find('div', class_='ca').find('a').get_text(strip=True)
        descricao = edital_div.find('div', class_='ca').find('a')['title']
        link_pagina_edital = edital_div['data-url']

        editais.append({
            'Título': titulo,
            'Detalhes': descricao,
            'Link Página Edital': link_pagina_edital
        })

    return editais


def fetch_pdf_links(edital_url):
    """Acessa a página do edital e retorna uma lista de todos os PDFs encontrados."""
    response = requests.get(edital_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    pdf_links = [a['href'] for a in soup.find_all('a', href=lambda href: href and href.endswith('.pdf'))]

    return pdf_links if pdf_links else ["Nenhum PDF encontrado"]


editais_info = fetch_edital_links(BASE_URL)

# Extrai o link do PDF para cada edital
for edital in editais_info:
    pdfs = fetch_pdf_links(edital['Link Página Edital'])
    edital['PDFs'] = pdfs

# Converte para DataFrame e salva como CSV
df_editais = pd.DataFrame(editais_info)
df_editais.to_csv("editais_concursos.csv", index=False, encoding='utf-8')

print(df_editais.head())
