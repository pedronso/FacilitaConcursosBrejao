import os
import json
import re
from tests_vars import dict_models
from re import sub, DOTALL

BASE_RESULTS_DIR = "data/responses"

def save_results(config_name: str, perguntas_respostas_dict: dict):
    """
    Salva os resultados do modelo em um arquivo JSON, organizando-os por configuração.

    Args:
        config_name (str): Nome da configuração (normalmente o nome da pasta).
        perguntas_respostas_dict (dict): Dicionário contendo perguntas e respostas.
    """
    config_results_path = os.path.join(BASE_RESULTS_DIR, f"{config_name}_resultados.json")

    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

    # Carregar resultados existentes (se houver)
    try:
        with open(config_results_path, 'r', encoding='utf-8') as file:
            resultados = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        resultados = {}

    # Criar índice de teste (para controle de versões dos testes)
    teste_index = len(resultados.keys())
    teste_name = f'teste_{teste_index}'

    model_data = dict_models.copy()

    # Remover tags desnecessárias nas respostas (como "<think>...</think>")
    for key, value in perguntas_respostas_dict.items():
        if isinstance(value, str):
            perguntas_respostas_dict[key] = sub(r"<think>.*?</think>", "", value, flags=DOTALL)

    model_data['config_name'] = config_name  # Nome da configuração atual
    model_data['embedding_model'] = config_name  # Define o modelo de embeddings conforme a pasta
    model_data['respostas'] = perguntas_respostas_dict

    resultados[teste_name] = model_data

    # Salvar os resultados atualizados no arquivo JSON
    with open(config_results_path, 'w', encoding='utf-8') as file:
        json.dump(resultados, file, indent=4, ensure_ascii=False)

    print(f"✅ Resultados salvos em: {config_results_path}")
    
#exemplo de uso:
#save_results(config_name="DeepSeek_GTE-Large_200_40_ON_ON_ON", perguntas_respostas_dict=respostas)
