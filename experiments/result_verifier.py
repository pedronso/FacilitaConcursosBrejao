from models.llm_model import LLMReviewerModel
import json

RESULT_PATH = "data/responses/resultados.json"

with open(RESULT_PATH, 'r', encoding='utf-8') as file:
    resultados = json.load(file)

testes_reviews = {

}

llm_reviewer = LLMReviewerModel()

for key, value in resultados.items():
    print(value['respostas'])
    for _key in value['respostas']:
        prompt = f'Analise esta resposta: {_key}\nresposta: {value['respostas'][key]}'
        response = llm_reviewer.generate_response(prompt)
        print(f'pergunta: {_key}\nNota: {response}')
    
    """
    testes_reviews[key] = {
        'ai_model': value['ai_model'],
        'embedding_model': value['embedding_model'],
        "chunk_size": value['chunk_size'],
        "topk": value['topk']
    }
    """