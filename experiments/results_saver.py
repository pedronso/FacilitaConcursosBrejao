from tests_vars import dict_models
import json

RESULT_PATH = "data/responses/resultados.json"

def save_results(perguntas_respostas_dict: dict):
    try:
        with open(RESULT_PATH, 'r', encoding='utf-8') as file:
            resultados = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        resultados = {} 
    
    teste_index = len(resultados.keys())
    teste_name = f'teste_{teste_index}'
    model_data = dict_models.copy()
    model_data['respostas'] = perguntas_respostas_dict
    model_data['avaliacoes'] = {}
    resultados[teste_name] = model_data

    with open(RESULT_PATH, 'w', encoding='utf-8') as file:
        json.dump(resultados, file, indent=4, ensure_ascii=False)


