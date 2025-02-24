import os
import json
from models.llm_model import LLMReviewerModel

# Diretórios de entrada e saída
RESULTS_DIR = "data/processed/respostas_utf8"
METRICS_DIR = "data/processed/metricas"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

class ResultVerifier:
    def __init__(self):
        self.llm_reviewer = LLMReviewerModel()

    def save_json(self, data, filepath):
        """Salva um dicionário como JSON no local especificado."""
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

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

                    prompt = f"Avalie a seguinte resposta:\n\n**Pergunta:** {pergunta}\n**Resposta:** {resposta}"
                    avaliacao = self.llm_reviewer.generate_response(prompt)

                    # Extrai apenas o número da nota e garante que está no intervalo [0, 10]
                    nota = "".join([c for c in avaliacao if c.isdigit()])
                    nota = int(nota) if nota else 0
                    nota = max(0, min(nota, 10))  # Garante que esteja entre 0 e 10
                    
                    avaliacoes[pergunta] = nota
                    media_total += nota
                    total_avaliacoes += 1
                    print(f"\nNota ajustada: {nota}")

                # Calcula a média geral
                media_final = media_total / total_avaliacoes if total_avaliacoes > 0 else 0
                print(f"\n✅✅✅ Média FINAL: {media_final:.2f}")

                # Salva as métricas no arquivo correspondente
                resultado_final = {"avaliacoes": avaliacoes, "media": media_final}
                self.save_json(resultado_final, metric_filepath)

                print(f"📊 Arquivo avaliado: {filename} | Média: {media_final:.2f}")
                print(f"📁 Métricas salvas em: {metric_filepath}")

    def corrigir_media_arquivo(self, filepath):
        """Recalcula e corrige a média final de um arquivo específico de métricas."""
        if not os.path.exists(filepath):
            print(f"❌ Arquivo não encontrado: {filepath}")
            return

        print(f"\n📊 Corrigindo média de {filepath}...")

        with open(filepath, "r", encoding="utf-8") as file:
            metricas = json.load(file)

        avaliacoes = metricas.get("avaliacoes", {})

        # Recalcula a média apenas com os valores existentes
        total_notas = sum(avaliacoes.values())
        num_avaliacoes = len(avaliacoes)

        media_corrigida = total_notas / num_avaliacoes if num_avaliacoes > 0 else 0
        media_corrigida = round(media_corrigida, 2)  # Mantém duas casas decimais

        # Atualiza a média corrigida no JSON
        metricas["media"] = media_corrigida

        # Salva o arquivo corrigido
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(metricas, file, indent=4, ensure_ascii=False)

        print(f"✅ Média corrigida: {media_corrigida:.2f} | Arquivo atualizado: {filepath}")

    def corrigir_todas_as_medias(self):
        """Percorre e corrige a média de todos os arquivos de métricas na pasta."""
        print("\n📊 Corrigindo médias em todos os arquivos de métricas...")
        for filename in sorted(os.listdir(METRICS_DIR)):
            if filename.endswith("_metricas.json"):
                metric_filepath = os.path.join(METRICS_DIR, filename)
                self.corrigir_media_arquivo(metric_filepath)

        print("\n✅ Todas as médias foram corrigidas!")


def avaliar_respostas():
    """Executa a avaliação das respostas geradas pela RAG."""
    verifier = ResultVerifier()
    print("\n🔍 Avaliando respostas da estrutura atual...")
    verifier.review_new_structure()
    print("\n✅ Avaliação de respostas concluída!")


if __name__ == "__main__":
    avaliar_respostas()
