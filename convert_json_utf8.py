import os
import json

INPUT_DIR = "data/processed/respostas/"
OUTPUT_DIR = "data/processed/respostas_utf8/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_json_to_utf8(input_path, output_path):
    """Lê um JSON, reescreve em UTF-8 sem caracteres escapados e salva em um novo local."""
    try:
        with open(input_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print(f"✅ Convertido: {input_path} → {output_path}")
    except Exception as e:
        print(f"❌ Erro ao processar {input_path}: {e}")

def process_all_json_files():
    """Percorre todos os arquivos JSON na pasta INPUT_DIR e converte para UTF-8 na OUTPUT_DIR."""
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".json"):
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename)
            convert_json_to_utf8(input_path, output_path)

if __name__ == "__main__":
    process_all_json_files()
