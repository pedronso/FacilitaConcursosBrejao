from langchain_groq import ChatGroq
import os
import json
import requests
from dotenv import load_dotenv
import tests_vars

load_dotenv()

class LLMModel:
    def __init__(self, model_name=tests_vars.dict_models['ai_model']):
        """
        Inicializa a conexão com o modelo LLM da Groq.
        - model_name: Modelo a ser utilizado (ex: `llama3-70b-8192` ou `mixtral-8x7b-32768`).
        """
        self.llm = ChatGroq(
            model=model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
            max_tokens=2000, 
            timeout=None,
            max_retries=2
        )

    def generate_response(self, prompt):
        system = """Você é um assistente especializado em concursos públicos.
Responda somente em português.
Seja direto.
Se não souber, apenas diga que não sabe.
Caso a informação venha repetida, apenas cite ela uma vez.
Fale somente sobre assuntos relacionados a concursos e estudos.
"""
        messages = [('system', system),
                    ("human", prompt)]
        try:
            resposta = self.llm.invoke(messages).content
        except Exception as e:
            print(f"Erro ao processar: {e}")

        return resposta


class LocalLLMModel:
    def __init__(self, model_name= "llama3.2:latest", url= "http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.url = url
        

    def generate_response(self, prompt):
        payload = {
            'model' : self.model_name,
            'system': """Você é um especialista em concursos públicos, ajude a responder as dúvidas, e caso não sabia, não responda.""",
            'prompt': prompt,
        }
        llm_response = ''

        with requests.post("http://localhost:11434/api/generate", json=payload, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    #print(data.get("response", ""), end="", flush=True)
                    llm_response = llm_response + data.get("response", "")
        #print(llm_response)
        return llm_response
        


class LLMReviewerModel(LLMModel):
    
    def generate_response(self, prompt):
        system = """Você é um avaliador de respostas. Sua tarefa é analisar a qualidade de uma resposta com base na pergunta feita e atribuir uma nota de 0 a 10, considerando os seguintes critérios:

1. **Precisão**: A resposta está correta e alinhada com a pergunta?
2. **Completude**: A resposta fornece todas as informações necessárias?
3. **Relevância**: A resposta é relevante para a pergunta feita?
4. **Clareza**: A resposta é clara e bem estruturada?

Instruções:
- Atribua uma nota de 0 a 10 para a resposta.
- Responda **APENAS COM O NÚMERO DA NOTA**, sem justificativas, explicações ou texto adicional.

Exemplos:
Pergunta: "Qual é a data da inscrição para o concurso da Aeronáutica?"
Resposta: "O período de inscrição para o concurso da Aeronáutica (CFS 1/2026) é de 15/01/2025 a 14/02/2025."
Nota: 10

Pergunta: "Qual é o salário para o concurso da Aeronáutica?"
Resposta: "O texto fornecido não especifica o salário para o concurso da Aeronáutica. Para obter informações sobre o salário, é recomendado consultar o site oficial da Aeronáutica ou entrar em contato com o Serviço de Recrutamento e Preparo de Pessoal da Aeronáutica (SEREP) pelos telefones fornecidos no texto."
Nota: 2

Pergunta: "Qual é a data da inscrição para o concurso da Aeronáutica?"
Resposta: "Não sei."
Nota: 0
"""
        messages = [('system', system),
                    ("human", prompt)]
        try:
            resposta = self.llm.invoke(messages).content
        except Exception as e:
            print(f"Erro ao processar: {e}")

        return resposta




# Teste
if __name__ == "__main__":
    model = LLMModel()
    resposta = model.generate_response("Quais são os concursos públicos com inscrições abertas?")
    print(f"Resposta do modelo: {resposta}")
