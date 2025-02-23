import os
from models.llm_model import LLMReviewerModel
import json


RESULT_PATH = "data/responses/resultados.json"

#new
RESULTS_DIR = "data/processed/respostas"
METRICS_DIR = "data/processed/metricas"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

class ResultVerifier:
    def __init__(self):
        self.llm_reviewer = LLMReviewerModel()


    def save_json(self, dict, local='resultados'):
        json_string = json.dumps(dict, indent=4, ensure_ascii=False)
        with open(f'data/responses/{local}.json', 'w', encoding='utf-8') as arquivo:
            arquivo.write(json_string)


    def review(self) -> None:
        with open(RESULT_PATH, 'r', encoding='utf-8') as file:
            resultados = json.load(file)
        media = 0
        total = 0

        for key, value in resultados.items():
            #print(value)
            if not isinstance(value, dict):
                continue
            for _key in value['respostas']:
                #formatando o prompt

                if not 'avaliacoes' in value:
                    value['avaliacoes'] = {}
                
                if _key in value['avaliacoes']:
                    print('nota dada, pulando')
                    continue
                prompt = f'''avalie a seguinte pergunta e resposta:
        Pergunta: {_key}
        Resposta: {value['respostas'][_key]}'''
                
                response = self.llm_reviewer.generate_response(prompt)

                #tirar apenas o número da nota
                nota = ''
                for letter in response:
                    if letter in '0123456789':
                        nota += letter
                
                if not 'avaliacoes' in value:
                    value['avaliacoes'] = {}
                
                value['avaliacoes'][_key] = int(nota)
                print(f'teste:{key}\npergunta: {_key}\n{int(nota)}')
                media += int(nota)
                total += 1
                value['media'] = media/total     
                self.save_json(resultados)

        
    #new: processa data/processed/respostas/
    def review_new_structure(self):
        """Avalia as respostas da nova estrutura (`data/processed/respostas/`) e salva em `data/processed/metricas/`."""
        for filename in sorted(os.listdir(RESULTS_DIR)):
            if filename.endswith("_respostas.json"):
                filepath = os.path.join(RESULTS_DIR, filename)
                print(f"\n📈 Avaliando {filename}...")

                with open(filepath, 'r', encoding='utf-8') as file:
                    respostas = json.load(file)

                media_total = 0
                total_avaliacoes = 0
                avaliacoes = {}

                for pergunta, resposta in respostas.items():
                    print(f"🔍 Avaliando resposta: {pergunta}")

                    prompt = f"Avalie a seguinte pergunta e resposta:\nPergunta: {pergunta}\nResposta: {resposta}"
                    avaliacao = self.llm_reviewer.generate_response(prompt)

                    # extrai nota
                    nota = "".join([c for c in avaliacao if c.isdigit()])
                    nota = int(nota) if nota else 0
                    avaliacoes[pergunta] = nota
                    media_total += nota
                    total_avaliacoes += 1

                media_final = media_total / total_avaliacoes if total_avaliacoes > 0 else 0

                # cria json final
                resultado_final = {"avaliacoes": avaliacoes, "media": media_final}
                output_path = os.path.join(METRICS_DIR, filename.replace("_respostas.json", "_metricas.json"))
                self.save_json(resultado_final, output_path)

                print(f"📊 Arquivo avaliado: {filename} | Média: {media_final}")
                print(f"📁 Métricas salvas em: {output_path}")

def avaliar_respostas():
    """Executa a avaliação das respostas antigas e novas."""
    verifier = ResultVerifier()

    print("\n🔍 Avaliando respostas no formato antigo...")
    verifier.review()

    print("\n🔍 Avaliando respostas no formato novo...")
    verifier.review_new_structure()

    print("\n✅ Avaliação completa!")

if __name__ == "__main__":
    avaliar_respostas()
    
    
#print(resultados['teste1']['avaliacoes'])