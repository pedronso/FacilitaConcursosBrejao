import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Diretórios
TIMES_FILE = "data/processed/metricas/media/tempos_resumo.json"
METRICS_FILE = "data/processed/metricas/media/metricas_resumo.json"
GRAPHICS_DIR = "data/processed/metricas/media/graficos/"

# Criar diretório de gráficos se não existir
os.makedirs(GRAPHICS_DIR, exist_ok=True)

# Aplicando um estilo mais moderno ao seaborn
sns.set_style("whitegrid")


def gerar_grafico_tempo_medio():
    """Gera um gráfico comparando o tempo médio de resposta entre os modelos."""
    with open(TIMES_FILE, "r", encoding="utf-8") as f:
        tempos = json.load(f)

    # Extraindo médias gerais dos modelos
    df_media = pd.DataFrame(tempos["Media_Geral"].items(), columns=["Modelo", "Tempo Médio de Resposta"])

    # 📊 Gráfico comparativo dos tempos médios entre os modelos
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df_media, x="Modelo", y="Tempo Médio de Resposta", palette="Blues_d")

    # Adicionando valores sobre as barras
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}s", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=12, color='black')

    plt.xlabel("Modelo", fontsize=12)
    plt.ylabel("Tempo Médio de Resposta (s)", fontsize=12)
    plt.title("Comparação do Tempo Médio de Resposta entre Modelos", fontsize=14, fontweight='bold')
    plt.ylim(0, df_media["Tempo Médio de Resposta"].max() + 5)

    # Salvar o gráfico
    output_path = os.path.join(GRAPHICS_DIR, "comparacao_tempo_medio_modelos.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Gráfico salvo em: {output_path}")


def gerar_grafico_media_notas_por_llm():
    """Gera um gráfico comparando a média das notas entre os modelos."""
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        metricas = json.load(f)

    # Criar um DataFrame a partir do JSON
    df_metricas = pd.DataFrame(metricas)

    # Criar uma coluna para identificar se a configuração pertence ao DeepSeek ou LLaMA-3
    df_metricas["Modelo"] = df_metricas["Configuracao"].apply(lambda x: "DeepSeek" if "DeepSeek" in x else "LLaMA-3")

    # Calcular a média das notas por LLM
    df_media_por_llm = df_metricas.groupby("Modelo")["Media"].mean().reset_index()

    # 📊 Gráfico comparativo da média das notas entre os modelos
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df_media_por_llm, x="Modelo", y="Media", palette="Greens_d")

    # Adicionando valores sobre as barras
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=12, color='black')

    plt.xlabel("Modelo", fontsize=12)
    plt.ylabel("Média das Notas", fontsize=12)
    plt.title("Comparação da Média das Notas entre Modelos LLM", fontsize=14, fontweight='bold')
    plt.ylim(0, df_media_por_llm["Media"].max() + 2)

    # Salvar o gráfico
    output_path = os.path.join(GRAPHICS_DIR, "comparacao_media_notas_modelos.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Gráfico salvo em: {output_path}")


def gerar_grafico_media_notas_por_embedding():
    """Gera um gráfico comparando a média das notas entre os modelos de embedding."""
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        metricas = json.load(f)

    # Criar um DataFrame a partir do JSON
    df_metricas = pd.DataFrame(metricas)

    # Criar uma coluna para identificar o modelo de embedding usado (GTE-Large ou E5-Large)
    df_metricas["Embedding"] = df_metricas["Configuracao"].apply(
        lambda x: "GTE-Large" if "GTE-Large" in x else "E5-Large"
    )

    # Calcular a média das notas por tipo de embedding
    df_media_por_embedding = df_metricas.groupby("Embedding")["Media"].mean().reset_index()

    # 📊 Gráfico comparativo da média das notas por embedding
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df_media_por_embedding, x="Embedding", y="Media", palette="Purples_d")

    # Adicionando valores sobre as barras
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=12, color='black')

    plt.xlabel("Modelo de Embedding", fontsize=12)
    plt.ylabel("Média das Notas", fontsize=12)
    plt.title("Comparação da Média das Notas por Modelo de Embedding", fontsize=14, fontweight='bold')
    plt.ylim(0, df_media_por_embedding["Media"].max() + 2)

    # Salvar o gráfico
    output_path = os.path.join(GRAPHICS_DIR, "comparacao_media_notas_embeddings.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Gráfico salvo em: {output_path}")


if __name__ == "__main__":
    gerar_grafico_tempo_medio()
    gerar_grafico_media_notas_por_embedding()
    gerar_grafico_media_notas_por_llm()
