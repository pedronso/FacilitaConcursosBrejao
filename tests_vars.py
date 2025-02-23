
# thenlper/gte-large quase | bom
# sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | mais ou menos
# intfloat/multilingual-e5-base | ruim
# intfloat/e5-large-v2 | bom


# llama3-70b-8192
# deepseek-r1-distill-llama-70b
# mixtral-8x7b-32768

dict_models = {
    'ai_model': 'llama3-70b-8192',
    'embedding_model': 'thenlper/gte-large', 
    'labeled': False,
    'normalized': False,  
    'stop-word': False,  
    'chunk_size': 200,  
    'chunk_overlap': 0, 
    'topk': 15 
}

# Fallbacks para evitar erros caso alguma chave falte
DEFAULTS = {
    'ai_model': 'llama3-70b-8192',
    'embedding_model': 'thenlper/gte-large',
    'labeled': False,
    'normalized': False,
    'stop-word': False,
    'chunk_size': 200,
    'chunk_overlap': 0,
    'topk': 15
}

def get_model_var(key):
    return dict_models.get(key, DEFAULTS.get(key))

idx = []

def usar_chunking_text(texto,a,b,c,d,e):
    from pipelines.extractor import chunking_texto
    return chunking_texto(texto,a,b,c,d,e)

def process_indexes(config_name:str) -> None:
    idx = []
    config = config_name.lower().split('_')[2:]
    chunk_size = int(config[0])
    chunk_overlap = int(config[1])
    label= True if config[2] == 'on' else False
    normalization= True if config[3] == 'on' else False
    stop_word= True if config[4] == 'on' else False
    

    print(f'config test vars: {config}')

    """Extrai texto dos PDFs e divide em chunks."""
    print("\n📄 [2/6] Extraindo textos dos PDFs...")
    #resultados = processar_downloads_e_extração(CSV_EDITAIS, "data/raw/")
    #resultados = chunking_texto("data/raw/text.txt")
    files = [
        "data/raw/ibge.txt",
    ]
    files = [
        "data/extracted_pedro/aeb.txt",
        "data/extracted_pedro/aeronautica.txt",
        "data/extracted_pedro/cceb.txt",
        "data/extracted_pedro/cnen.txt",
        "data/extracted_pedro/funai.txt",
        "data/extracted_pedro/ibama.txt",
        "data/extracted_pedro/ibge.txt",
        "data/extracted_pedro/marinha.txt",
        "data/extracted_pedro/mpu.txt",
        "data/extracted_pedro/trf.txt",
    ]

    all_chunks = []
    print('processando índices')
    for file in files:
        chunks = usar_chunking_text(file, label, normalization, stop_word, chunk_size, chunk_overlap)
        index = len(all_chunks) + 1
        print(index)
        idx.append(index)
        all_chunks.extend(chunks)
