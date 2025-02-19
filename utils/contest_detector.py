import re

CONTEST_ALIASES = {
    "IBGE": ["ibge", "instituto brasileiro de geografia e estatística"],
    "FUNAI": ["funai", "fundação nacional do índio", "fundação nacional dos povos indígenas"],
    "IBAMA": ["ibama", "instituto brasileiro do meio ambiente", "meio ambiente"],
    "MARINHA": ["marinha", "marinha do brasil", "força naval"],
    "AERONÁUTICA": ["aeronáutica", "força aérea brasileira", "fab"],
    "CNEN": ["cnen", "comissão nacional de energia nuclear", "energia nuclear"],
    "TRF": ["trf", "tribunal regional federal"]
}

def detect_contest(query):
    """
    Identifica se há menção a um concurso na query e retorna o nome padronizado.

    Args:
    - query (str): Pergunta do usuário.

    Returns:
    - str: Nome padronizado do concurso (ex: "IBGE", "FUNAI") ou None se não encontrar.
    """
    query = query.lower().strip()  # Normaliza a entrada

    # Itera sobre o dicionário de sinônimos
    for contest, aliases in CONTEST_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", query):  # Usa regex para busca exata
                return contest  # Retorna o nome padronizado do concurso

    return None  
