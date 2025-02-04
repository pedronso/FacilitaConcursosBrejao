from langchain_groq import ChatGroq
import os
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
        
        return " ".join(responses)  # 🔹 Junta todas as respostas



# Teste
if __name__ == "__main__":
    model = LLMModel()
    resposta = model.generate_response("Quais são os concursos públicos com inscrições abertas?")
    print(f"Resposta do modelo: {resposta}")
