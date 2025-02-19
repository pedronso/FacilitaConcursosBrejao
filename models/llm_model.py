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
        context = "IBGE - ANEXO I – Quadro de Vagas e Postos de Inscrição {‘Função’: ‘APM’, ‘UF’: ‘ES’, ‘Município’: ‘Cariacica’, ‘VAGAS’: ‘2’, ‘AC’: ‘1’, ‘PPP’: ‘1’, ‘PcD’: ‘0’, ‘Endereço para inscrições’: ‘Av. Nossa Sra. dos Navegantes, 675 (Edifício Palácio do Café), 9º andar - Enseada do Suá. Vitória/ES’}, {‘Função’: ‘APM’, ‘UF’: ‘ES’, ‘Município’: ‘Vitória’, ‘VAGAS’: ‘12’, ‘AC’: ‘8’, ‘PPP’: ‘2’, ‘PcD’: ‘2’, ‘Endereço para inscrições’: ‘Av. Nossa Sra. dos Navegantes, 675 (Edifício Palácio do Café), 9º andar - Enseada do Suá. Vitória/ES’}, {‘Função’: ‘APM’, ‘UF’: ‘MG’, ‘Município’: ‘Iturama’, ‘VAGAS’: ‘2’, ‘AC’: ‘1’, ‘PPP’: ‘1’, ‘PcD’: ‘0’, ‘Endereço para inscrições’: ‘Rua Armando Fratari, 867, Vila Olímpica. Iturama/MG’}, {‘Função’: ‘APM’, ‘UF’: ‘MS’, ‘Município’: ‘Campo Grande’, ‘VAGAS’: ‘7’, ‘AC’: ‘6’, ‘PPP’: ‘1’, ‘PcD’: ‘0’, ‘Endereço para inscrições’: ‘Rua Barão do Rio Branco, 1431, Centro. Campo Grande/MS’}, {‘Função’: ‘APM’, ‘UF’: ‘SC’, ‘Município’: ‘Concórdia’, ‘VAGAS’: ‘5’, ‘AC’: ‘4’, ‘PPP’: ‘1’, ‘PcD’: ‘0’, ‘Endereço para inscrições’: ‘Rua Marechal Deodoro, 772, Centro. Concórdia/SC’}, {‘Função’: ‘APM’, ‘UF’: ‘SC’, ‘Município’: ‘Rio do Sul’, ‘VAGAS’: ‘1’, ‘AC’: ‘1’, ‘PPP’: ‘0’, ‘PcD’: ‘0’, ‘Endereço para inscrições’: ‘Rua Tuiuti, 20, salas 401 e 402, Centro. Rio do Sul/SC’}, {‘Função’: ‘SCQ’, ‘UF’: ‘SC’, ‘Município’: ‘Palmitos’, ‘VAGAS’: ‘1’, ‘AC’: ‘1’, ‘PPP’: ‘0’, ‘PcD’: ‘0’, ‘Endereço para inscrições’: ‘Rua Visconde de Rio Branco, 932, sala 102, Centro. Palmitos/SC’}"
        promp1 = f"Baseando-se somente nos seguintes textos:{context}\n\n responda: Quais são os quadros de vagas e postos de inscrição para o ibge?"
        system = """Você é um assistente especializado em concursos públicos e estudos para provas.  
Responda **apenas em português** e de forma **clara e objetiva**.  

📌 **Regras de resposta:**  
- Se a pergunta não estiver relacionada a concursos ou estudos, diga que não pode ajudar.  
- Se não souber a resposta, apenas diga **"Não tenho essa informação."**  
- Se houver repetição de informações no contexto, mencione cada item **apenas uma vez**.  
- Se houver leis, artigos ou regras nos editais, priorize a informação mais recente.  
- Sempre que possível, explique **com base no edital e em regras oficiais**.  

🔎 **Exemplo de comportamento esperado:**  
✅ **Usuário:** "Quantas vagas há para o cargo X?"  
✅ **Assistente:** "O edital informa que há 30 vagas para o cargo X."  

🚫 **Usuário:** "Me fale sobre esportes."  
🚫 **Assistente:** "Eu sou especializado apenas em concursos públicos e estudos."  
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
    def __init__(self, model_name='llama3-70b-8192'):
        """
        Inicializa a conexão com o modelo LLM da Groq.
        - model_name: Modelo a ser utilizado (ex: `llama3-70b-8192` ou `mixtral-8x7b-32768`).
        """
        super().__init__(model_name)

    def generate_response(self, prompt):
        system = """Você é um avaliador de respostas. Sua tarefa é analisar a qualidade de uma resposta com base na pergunta feita e atribuir uma nota de 0 a 10, considerando os seguintes critérios:

1. **Precisão**: A resposta está correta e alinhada com a pergunta?
2. **Completude**: A resposta fornece todas as informações necessárias?
3. **Relevância**: A resposta é relevante para a pergunta feita?
4. **Clareza**: A resposta é clara e bem estruturada?

### **Instruções**
- Atribua uma nota de **0 a 10** para a resposta.
- Responda **APENAS COM O NÚMERO DA NOTA**, sem justificativas, explicações ou texto adicional.
- Penalize respostas vagas, evasivas ou que indicam falta de informação:
  - Se a resposta disser "Não sei" ou não fornecer qualquer dado útil, a nota deve ser **0**.
  - Se a resposta for genérica e não agregar valor, a nota deve ser no máximo **2**.
  - Respostas parcialmente corretas, mas com falta de detalhes essenciais, devem ter notas entre **3 e 6**.
  - Apenas respostas completas e corretas devem receber **8 a 10**.

### **Exemplos**
Pergunta: "Qual é a data da inscrição para o concurso da Aeronáutica?"
Resposta: "O período de inscrição para o concurso da Aeronáutica (CFS 1/2026) é de 15/01/2025 a 14/02/2025."
Nota: 10

Pergunta: "Qual é o salário para o concurso da Aeronáutica?"
Resposta: "O texto fornecido não especifica o salário para o concurso da Aeronáutica. Para obter informações sobre o salário, é recomendado consultar o site oficial da Aeronáutica ou entrar em contato com o Serviço de Recrutamento e Preparo de Pessoal da Aeronáutica (SEREP) pelos telefones fornecidos no texto."
Nota: 2

Pergunta: "Qual é a data da inscrição para o concurso da Aeronáutica?"
Resposta: "Não sei."
Nota: 0

Pergunta: "Quais são os requisitos para o concurso?"
Resposta: "Os requisitos variam conforme o edital. Recomenda-se verificar diretamente no site da Aeronáutica."
Nota: 1
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
