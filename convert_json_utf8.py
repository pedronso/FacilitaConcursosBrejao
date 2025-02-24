import os
import json
import re

INPUT_DIR = "data/processed/respostas/"
OUTPUT_DIR = "data/processed/respostas_utf8/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Padrões que identificam os modelos no início da resposta
MODEL_TAGS = [
    r"^\[Model used: deepseek-r1-distill-llama-70b\]\s*",
    r"^\[Model used: llama3-8b-8192\]\s*",
    r"^\[Model used: llama3-70b-8192\]\s*",
]

def clean_model_prefix(text):
    """Remove o prefixo '[Model used: ...]' caso esteja no início da resposta."""
    for pattern in MODEL_TAGS:
        text = re.sub(pattern, "", text).strip()
    return text

def convert_json_to_utf8_and_clean(input_path, output_path):
    """Lê um JSON, remove prefixos '[Model used: ...]', reescreve em UTF-8 e salva."""
    try:
        with open(input_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Processar cada resposta para remover o prefixo
        cleaned_data = {question: clean_model_prefix(answer) for question, answer in data.items()}

        # Salvar no novo arquivo com encoding UTF-8 sem caracteres escapados
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(cleaned_data, file, ensure_ascii=False, indent=4)

        print(f"✅ Processado e salvo: {input_path} → {output_path}")
    except Exception as e:
        print(f"❌ Erro ao processar {input_path}: {e}")

def process_all_json_files():
    """Percorre todos os arquivos JSON na pasta INPUT_DIR e processa para UTF-8 na OUTPUT_DIR."""
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".json"):
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename)
            convert_json_to_utf8_and_clean(input_path, output_path)

if __name__ == "__main__":
    process_all_json_files()
