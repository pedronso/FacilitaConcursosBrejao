import os
import json
from models.llm_model import LLMReviewerModel

# Diretórios de entrada e saída
RESULTS_DIR = "data/processed/respostas"
METRICS_DIR = "data/processed/metricas"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

class ResultVerifier:
    def __init__(self):
        self.llm_reviewer = LLMReviewerModel()

    def save_json(self, data, filepath):
        """Salva um dicionário como JSON no local especificado."""
        json_string = json.dumps(data, indent=4, ensure_ascii=False)
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(json_string)

    def review_new_structure(self):
        """Avalia todas as respostas da nova estrutura e salva as métricas."""

        for filename in sorted(os.listdir(RESULTS_DIR)):
            if filename.endswith("_respostas.json"):
                filepath = os.path.join(RESULTS_DIR, filename)
                metric_filepath = os.path.join(METRICS_DIR, filename.replace("_respostas.json", "_metricas.json"))

                # Se já foi avaliado, pula
                if os.path.exists(metric_filepath):
                    print(f"🔹 {filename} já foi avaliado, pulando...")
                    continue

                print(f"\n📈 Avaliando respostas de {filename}...")

                with open(filepath, "r", encoding="utf-8") as file:
                    respostas = json.load(file)

                media_total = 0
                total_avaliacoes = 0
                avaliacoes = {}

                for pergunta, resposta in respostas.items():
                    print(f"🔍 Avaliando resposta: {pergunta}")

                    prompt = f"Avalie a seguinte pergunta e resposta:\n\n**Pergunta:** {pergunta}\n**Resposta:** {resposta}"
                    avaliacao = self.llm_reviewer.generate_response(prompt)

                    # Extrai apenas o número da nota
                    nota = "".join([c for c in avaliacao if c.isdigit()])
                    nota = int(nota) if nota else 0
                    avaliacoes[pergunta] = nota
                    media_total += nota
                    total_avaliacoes += 1

                # Calcula a média geral
                media_final = media_total / total_avaliacoes if total_avaliacoes > 0 else 0

                # Salva as métricas no arquivo correspondente
                resultado_final = {"avaliacoes": avaliacoes, "media": media_final}
                self.save_json(resultado_final, metric_filepath)

                print(f"📊 Arquivo avaliado: {filename} | Média: {media_final}")
                print(f"📁 Métricas salvas em: {metric_filepath}")

def avaliar_respostas():
    """Executa a avaliação das respostas geradas pela RAG."""
    verifier = ResultVerifier()
    print("\n🔍 Avaliando respostas da estrutura atual...")
    verifier.review_new_structure()
    print("\n✅ Avaliação de respostas concluída!")

if __name__ == "__main__":
    avaliar_respostas()
