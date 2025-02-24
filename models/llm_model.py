from langchain_groq import ChatGroq
import os
import json
import requests
from dotenv import load_dotenv
import time

load_dotenv()

# Mapeamento de modelos com base no nome da pasta de configuração.
CONFIG_MODEL_MAP = {
    "DeepSeek": "deepseek-r1-distill-llama-70b",
    "LLaMA": "llama3-8b-8192", #"llama3-70b-8192",
    "Mixtral": "mixtral-8x7b-32768",
}

# Lista de modelos de fallback para usar caso o modelo atual falhe.
FALLBACK_MODELS = [
    "lama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "mixtral-8x7b-32768", 
]

from langchain_groq import ChatGroq
import os
import json
import requests
from dotenv import load_dotenv
import time
from collections import deque

load_dotenv()

# Mapeamento de modelos com base no nome da pasta de configuração.
CONFIG_MODEL_MAP = {
    "DeepSeek": "deepseek-r1-distill-llama-70b",
    "LLaMA": "llama3-8b-8192",
    "Mixtral": "mixtral-8x7b-32768",
}

# Lista de modelos de fallback para usar caso o modelo atual falhe.
FALLBACK_MODELS = [
    "lama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "mixtral-8x7b-32768", 
]

# Limites do modelo DeepSeek
REQUESTS_PER_MINUTE = 30
TOKENS_PER_MINUTE = 6000

# Controle de requisições por minuto
request_timestamps = deque()

class LLMModel:
    def __init__(self, config_name):
        """
        Inicializa a conexão com o modelo LLM da Groq com base no nome da configuração.
        Em caso de erro, tenta modelos de fallback.
        """
        self.config_name = config_name
        self.model_name = self.get_model_from_config(config_name)
        self.init_llm(self.model_name)

    def get_model_from_config(self, config_name):
        """
        Obtém o modelo correto com base no nome da pasta de configuração.
        Exemplo: "DeepSeek_GTE-Large_200_40_ON_ON_ON" → "deepseek-r1-distill-llama-70b"
        """
        for key, model in CONFIG_MODEL_MAP.items():
            if key.lower() in config_name.lower():
                return model
        return FALLBACK_MODELS[0]

    def init_llm(self, model_name):
        """Inicializa a API da Groq com o modelo especificado e armazena o modelo atual."""
        self.llm = ChatGroq(
            model=model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
            max_tokens=2000, 
            timeout=None,
            max_retries=2
        )
        self.current_model = model_name
        print(f"🔹 Modelo carregado: {model_name}")

    def wait_for_rate_limit(self):
        """Garante que não excedemos 30 requisições por minuto."""
        while len(request_timestamps) >= REQUESTS_PER_MINUTE:
            elapsed_time = time.time() - request_timestamps[0]
            if elapsed_time < 60:
                sleep_time = 60 - elapsed_time
                print(f"⏳ Esperando {sleep_time:.2f} segundos para respeitar o limite de requisições por minuto...")
                time.sleep(sleep_time)
            request_timestamps.popleft()

    def generate_response(self, prompt):
        """
        Gera uma resposta utilizando o modelo LLM.
        - **Se tokens do input excederem 6000**, reduz o input pela metade.
        - **Se atingir o rate limit de 30 requisições por minuto**, aguarda.
        - **Se ocorrer erro, tenta um fallback automático**.
        """
        wait_time = 30
        original_prompt = prompt  # Guarda o prompt original para ajustes futuros

        while True:
            self.wait_for_rate_limit()  # Garante que não excedemos o limite de requisições
            
            try:
                estimated_output_tokens = 1500  # Suposição inicial do tamanho da resposta
                total_tokens = len(prompt.split()) + estimated_output_tokens
                
                # Se ultrapassarmos os 6000 tokens permitidos, reduzimos o input pela metade
                while total_tokens > TOKENS_PER_MINUTE:
                    print("⚠️ Excesso de tokens! Reduzindo input pela metade...")
                    prompt = " ".join(prompt.split()[:len(prompt.split()) // 2])
                    total_tokens = len(prompt.split()) + estimated_output_tokens
                
                request_timestamps.append(time.time())  # Registra o timestamp da requisição
                
                response = self.llm.invoke([("system", "Responda com precisão."), ("human", prompt)]).content
                return f"[Model used: {self.current_model}] {response}"

            except Exception as e:
                error_msg = str(e)
                print(f"❌ Erro ao gerar resposta: {error_msg}")

                if "rate_limit_exceeded" in error_msg:
                    print(f"⏳ Esperando {wait_time} segundos devido a rate limit...")
                    time.sleep(wait_time)
                    wait_time += 30

                elif "max_tokens" in error_msg or "token limit exceeded" in error_msg or "context length exceeded" in error_msg:
                    print("⚠️ Excesso de tokens: reduzindo prompt pela metade e tentando novamente...")
                    prompt = " ".join(original_prompt.split()[:len(original_prompt.split()) // 2])

                else:
                    print("🔄 Tentando fallback para outro modelo...")
                    if FALLBACK_MODELS:
                        new_model = FALLBACK_MODELS.pop(0)
                        print(f"⚡ Mudando para modelo alternativo: {new_model}")
                        self.init_llm(new_model)
                    else:
                        return f"Erro ao gerar resposta: {error_msg}"
    
    def generate_response_old(self, prompt):
            context = "IBGE - ANEXO I – Quadro de Vagas e Postos de Inscrição {‘Função’: ‘APM’, ‘UF’: ‘ES’, ‘Município’: ‘Cariacica’, ‘VAGAS’: ‘2’, ‘AC’: ‘1’, ‘PPP’: ‘1’, ‘PcD’: ‘0’, ‘Endereço para inscrições’: ‘Av. Nossa Sra. dos Navegantes, 675 (Edifício Palácio do Café), 9º andar - Enseada do Suá. Vitória/ES’}, {‘Função’: ‘APM’, ‘UF’: ‘ES’, ‘Município’: ‘Vitória’, ‘VAGAS’: ‘12’, ‘AC’: ‘8’, ‘PPP’: ‘2’, ‘PcD’: ‘2’, ‘Endereço para inscrições’: ‘Av. Nossa Sra. dos Navegantes, 675 (Edifício Palácio do Café), 9º andar - Enseada do Suá. Vitória/ES’}, {‘Função’: ‘APM’, ‘UF’: ‘MG’, ‘Município’: ‘Iturama’, ‘VAGAS’: ‘2’, ‘AC’: ‘1’, ‘PPP’: ‘1’, ‘PcD’: ‘0’, ‘Endereço para inscrições’: ‘Rua Armando Fratari, 867, Vila Olímpica. Iturama/MG’}, {‘Função’: ‘APM’, ‘UF’: ‘MS’, ‘Município’: ‘Campo Grande’, ‘VAGAS’: ‘7’, ‘AC’: ‘6’, ‘PPP’: ‘1’, ‘PcD’: ‘0’, ‘Endereço para inscrições’: ‘Rua Barão do Rio Branco, 1431, Centro. Campo Grande/MS’}, {‘Função’: ‘APM’, ‘UF’: ‘SC’, ‘Município’: ‘Concórdia’, ‘VAGAS’: ‘5’, ‘AC’: ‘4’, ‘PPP’: ‘1’, ‘PcD’: ‘0’, ‘Endereço para inscrições’: ‘Rua Marechal Deodoro, 772, Centro. Concórdia/SC’}, {‘Função’: ‘APM’, ‘UF’: ‘SC’, ‘Município’: ‘Rio do Sul’, ‘VAGAS’: ‘1’, ‘AC’: ‘1’, ‘PPP’: ‘0’, ‘PcD’: ‘0’, ‘Endereço para inscrições’: ‘Rua Tuiuti, 20, salas 401 e 402, Centro. Rio do Sul/SC’}, {‘Função’: ‘SCQ’, ‘UF’: ‘SC’, ‘Município’: ‘Palmitos’, ‘VAGAS’: ‘1’, ‘AC’: ‘1’, ‘PPP’: ‘0’, ‘PcD’: ‘0’, ‘Endereço para inscrições’: ‘Rua Visconde de Rio Branco, 932, sala 102, Centro. Palmitos/SC’}"
            prompt1 = f"Baseando-se somente nos seguintes textos:{context}\n\n responda: Quais são os quadros de vagas e postos de inscrição para o ibge?"
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
    def __init__(self, model_name="llama3.2:latest", url="http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.url = url

    def generate_response(self, prompt):
        payload = {
            'model': self.model_name,
            'system': "Você é um especialista em concursos públicos, ajude a responder as dúvidas, e caso não saiba, não responda.",
            'prompt': prompt,
        }
        llm_response = ''
        with requests.post(self.url, json=payload, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    llm_response += data.get("response", "")
        return llm_response

class LLMReviewerModel(LLMModel):
    def __init__(self, config_name="llama3-8b-8192"):
        super().__init__(config_name)

    def generate_response(self, prompt):
        system = """Você é um avaliador de respostas. Sua tarefa é analisar a qualidade de uma resposta com base na pergunta feita e atribuir uma nota de 0 a 10, considerando os seguintes critérios:

1. **Precisão**: A resposta está correta e alinhada com a pergunta?
2. **Completude**: A resposta fornece todas as informações necessárias?
3. **Relevância**: A resposta é relevante para a pergunta feita?
4. **Clareza**: A resposta é clara e bem estruturada?

### **Instruções**
- Atribua uma nota de **0 a 10** para a resposta, exclusivamente.
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
        messages = [('system', system), ("human", prompt)]
        try:
            response = self.llm.invoke(messages).content
            return response
        except Exception as e:
            print(f"Erro ao processar: {e}")
            return f"Erro: {e}"

# Teste
if __name__ == "__main__":
    model = LLMModel("DeepSeek_GTE-Large_200_40_ON_ON_ON")
    resposta = model.generate_response("Quais são os concursos públicos com inscrições abertas?")
    print(f"Resposta do modelo: {resposta}")
