from langchain_groq import ChatGroq
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

class LLMModel:
    def __init__(self, model_name="mixtral-8x7b-32768"):
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
        """Gera uma resposta baseada no prompt fornecido, dividindo em partes menores."""
        """
        MAX_INPUT_TOKENS = 2000
        chunks = [prompt[i:i + MAX_INPUT_TOKENS] for i in range(0, len(prompt), MAX_INPUT_TOKENS)]

        responses = []
        for chunk in chunks:
            messages = [("system", "Você é um assistente especializado em concursos públicos."),
                        ("human", chunk)]
            try:
                response = self.llm.invoke(messages).content
                responses.append(response)
            except Exception as e:
                print(f"Erro ao processar chunk: {e}")
        """
        system = """Você é um assistente especializado em concursos públicos.
Responda somente em português.
Seja direto.
Se não souber, apenas diga que não sabe.
Caso a informação venha repetida, apenas cite ela uma vez.
Fale somente sobre assuntos relacionados a concursos e estudos.
"""
        #print(system)
        #print(system.strip())
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
        




# Teste
if __name__ == "__main__":
    model = LLMModel()
    resposta = model.generate_response("Quais são os concursos públicos com inscrições abertas?")
    print(f"Resposta do modelo: {resposta}")
