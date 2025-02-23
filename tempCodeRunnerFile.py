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
        #falta mpu
    ]

    all_chunks = []
    for file in files:
        chunks = chunking_texto(file)
        index = len(all_chunks) + 1
        print(index)
        tests_vars.idx.append(index)
        all_chunks.extend(chunks)