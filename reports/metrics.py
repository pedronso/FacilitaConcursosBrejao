import time
from pipelines.rag import RAGPipeline

rag = RAGPipeline()
queries = [
    "Quais são os concursos para engenheiros?",
    "Quais os prazos de inscrição?",
    "Tem algum concurso para nível médio?"
]

resultados = []
for query in queries:
    start_time = time.time()
    resposta = rag.generate_answer(query)
    tempo_execucao = time.time() - start_time

    resultados.append({"Query": query, "Tempo de Resposta": tempo_execucao, "Resposta": resposta})

# Salvar resultados
import pandas as pd
df_metricas = pd.DataFrame(resultados)
df_metricas.to_csv("reports/metricas.csv", index=False)

print("✅ Métricas geradas com sucesso.")
