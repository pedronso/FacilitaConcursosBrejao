from models.llm_model import LLMReviewerModel
import json

RESULT_PATH = "data/responses/resultados.json"

with open(RESULT_PATH, 'r', encoding='utf-8') as file:
    resultados = json.load(file)

def save_json(dict, local='resultados'):
    json_string = json.dumps(dict, indent=4, ensure_ascii=False)
    with open(f'data/responses/{local}.json', 'w', encoding='utf-8') as arquivo:
        arquivo.write(json_string)


def __init__(self):
    self.llm_reviewer = LLMReviewerModel()



for key, value in resultados.items():
    #print(value['respostas'])

    for _key in value['respostas']:
        #formatando o prompt
        if _key in value['avaliacoes']:
            print('nota dada, pulando')
            continue
        prompt = f'''avalie a seguinte pergunta e resposta:
Pergunta: {_key}
Resposta: {value['respostas'][_key]}'''
        
        response = llm_reviewer.generate_response(prompt)

        #tirar apenas o número da nota
        nota = ''
        for letter in response:
            if letter in '0123456789':
                nota += letter
        if not 'avaliacoes' in value:
            value['avaliacoes'] = {}
        
        value['avaliacoes'][_key] = int(nota)
        print(f'teste:{key}\npergunta: {_key}\n{int(nota)}')
        save_json(resultados)
    

#calcular média



#print(resultados['teste1']['avaliacoes'])
