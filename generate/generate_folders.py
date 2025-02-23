import json
import os

CONFIGURACOES = [
    {"ID": "001", "LLM": "DeepSeek", "Embedding": "GTE-Large", "Chunk": 200, "Overlap": 0, "Label": "ON", "Normalization": "ON", "Stopwords": "ON"},
    {"ID": "002", "LLM": "DeepSeek", "Embedding": "GTE-Large", "Chunk": 200, "Overlap": 0, "Label": "OFF", "Normalization": "OFF", "Stopwords": "OFF"},
    {"ID": "003", "LLM": "DeepSeek", "Embedding": "GTE-Large", "Chunk": 200, "Overlap": 40, "Label": "ON", "Normalization": "ON", "Stopwords": "ON"},
    {"ID": "004", "LLM": "DeepSeek", "Embedding": "GTE-Large", "Chunk": 200, "Overlap": 40, "Label": "OFF", "Normalization": "OFF", "Stopwords": "OFF"},
    {"ID": "005", "LLM": "DeepSeek", "Embedding": "E5-Large", "Chunk": 300, "Overlap": 0, "Label": "ON", "Normalization": "ON", "Stopwords": "ON"},
    {"ID": "006", "LLM": "DeepSeek", "Embedding": "E5-Large", "Chunk": 300, "Overlap": 0, "Label": "OFF", "Normalization": "OFF", "Stopwords": "OFF"},
    {"ID": "007", "LLM": "DeepSeek", "Embedding": "E5-Large", "Chunk": 300, "Overlap": 60, "Label": "ON", "Normalization": "ON", "Stopwords": "ON"},
    {"ID": "008", "LLM": "DeepSeek", "Embedding": "E5-Large", "Chunk": 300, "Overlap": 60, "Label": "OFF", "Normalization": "OFF", "Stopwords": "OFF"},
    {"ID": "009", "LLM": "DeepSeek", "Embedding": "GTE-Large", "Chunk": 400, "Overlap": 0, "Label": "ON", "Normalization": "ON", "Stopwords": "ON"},
    {"ID": "010", "LLM": "DeepSeek", "Embedding": "GTE-Large", "Chunk": 400, "Overlap": 0, "Label": "OFF", "Normalization": "OFF", "Stopwords": "OFF"},
    {"ID": "011", "LLM": "DeepSeek", "Embedding": "GTE-Large", "Chunk": 400, "Overlap": 80, "Label": "ON", "Normalization": "ON", "Stopwords": "ON"},
    {"ID": "012", "LLM": "DeepSeek", "Embedding": "GTE-Large", "Chunk": 400, "Overlap": 80, "Label": "OFF", "Normalization": "OFF", "Stopwords": "OFF"},
    {"ID": "013", "LLM": "LLaMA 3", "Embedding": "GTE-Large", "Chunk": 200, "Overlap": 0, "Label": "ON", "Normalization": "ON", "Stopwords": "ON"},
    {"ID": "014", "LLM": "LLaMA 3", "Embedding": "GTE-Large", "Chunk": 200, "Overlap": 0, "Label": "OFF", "Normalization": "OFF", "Stopwords": "OFF"},
    {"ID": "015", "LLM": "LLaMA 3", "Embedding": "GTE-Large", "Chunk": 200, "Overlap": 40, "Label": "ON", "Normalization": "ON", "Stopwords": "ON"},
    {"ID": "016", "LLM": "LLaMA 3", "Embedding": "GTE-Large", "Chunk": 200, "Overlap": 40, "Label": "OFF", "Normalization": "OFF", "Stopwords": "OFF"},
    {"ID": "017", "LLM": "LLaMA 3", "Embedding": "E5-Large", "Chunk": 300, "Overlap": 0, "Label": "ON", "Normalization": "ON", "Stopwords": "ON"},
    {"ID": "018", "LLM": "LLaMA 3", "Embedding": "E5-Large", "Chunk": 300, "Overlap": 0, "Label": "OFF", "Normalization": "OFF", "Stopwords": "OFF"},
    {"ID": "019", "LLM": "LLaMA 3", "Embedding": "E5-Large", "Chunk": 300, "Overlap": 60, "Label": "ON", "Normalization": "ON", "Stopwords": "ON"},
    {"ID": "020", "LLM": "LLaMA 3", "Embedding": "E5-Large", "Chunk": 300, "Overlap": 60, "Label": "OFF", "Normalization": "OFF", "Stopwords": "OFF"},
    {"ID": "021", "LLM": "LLaMA 3", "Embedding": "GTE-Large", "Chunk": 400, "Overlap": 0, "Label": "ON", "Normalization": "ON", "Stopwords": "ON"},
    {"ID": "022", "LLM": "LLaMA 3", "Embedding": "GTE-Large", "Chunk": 400, "Overlap": 0, "Label": "OFF", "Normalization": "OFF", "Stopwords": "OFF"},
    {"ID": "023", "LLM": "LLaMA 3", "Embedding": "GTE-Large", "Chunk": 400, "Overlap": 80, "Label": "ON", "Normalization": "ON", "Stopwords": "ON"},
    {"ID": "024", "LLM": "LLaMA 3", "Embedding": "GTE-Large", "Chunk": 400, "Overlap": 80, "Label": "OFF", "Normalization": "OFF", "Stopwords": "OFF"}
]

BASE_DIR = "data/processed/configs"

def criar_pastas():
    """Cria diretórios para armazenar os chunks e índices FAISS."""
    for config in CONFIGURACOES:
        nome_pasta = f"{config['LLM'].replace(' ', '-')}_{config['Embedding']}_{config['Chunk']}_{config['Overlap']}_{config['Label']}_{config['Normalization']}_{config['Stopwords']}"
        caminho = os.path.join(BASE_DIR, nome_pasta)

        if not os.path.exists(caminho):
            os.makedirs(caminho)
            print(f"📂 Criado: {caminho}")
            
            # Salvar as configurações dentro da pasta
            with open(os.path.join(caminho, "config.json"), "w", encoding='utf-8') as config_file:
                json.dump(config, config_file, indent=4)
        else:
            print(f"✅ Já existe: {caminho}")

if __name__ == "__main__":
    criar_pastas()
