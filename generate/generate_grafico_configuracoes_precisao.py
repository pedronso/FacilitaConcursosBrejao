import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Diretórios e arquivo de entrada
JSON_FILE = "data/processed/metricas/media/configuracoes_com_medias.json"
GRAPHICS_DIR = "data/processed/metricas/media/graficos/"

# Criar diretório de gráficos se não existir
os.makedirs(GRAPHICS_DIR, exist_ok=True)

def gerar_tabela_configuracoes():
    """Lê o JSON de configurações com médias e gera uma tabela com ID, LLM, Embedding e Média."""
    # Carrega os dados do arquivo JSON
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Cria um DataFrame com os dados
    df = pd.DataFrame(data)
    
    # Seleciona apenas as colunas desejadas: ID, LLM, Embedding e Media
    df = df[["ID", "LLM", "Embedding", "Media"]]
    
    # Ordena pelo ID (opcional)
    df.sort_values(by="ID", inplace=True)
    
    # Cria uma figura e adiciona uma tabela com os dados
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.35 + 2))  # ajusta o tamanho vertical conforme o número de linhas
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    
    plt.title("Tabela de Configurações e Média das Notas", fontsize=14)
    
    # Salva a tabela como imagem
    output_path = os.path.join(GRAPHICS_DIR, "tabela_configuracoes_medias.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Tabela salva em: {output_path}")

if __name__ == "__main__":
    gerar_tabela_configuracoes()
