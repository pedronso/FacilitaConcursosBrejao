# thenlper/gte-large quase | bom
# sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | mais ou menos
# intfloat/multilingual-e5-base | ruim
# intfloat/e5-large-v2 | 


# llama3-70b-8192
# deepseek-r1-distill-llama-70b
# mixtral-8x7b-32768

dict_models = {
    'ai_model' : 'mixtral-8x7b-32768',
    'embedding_model': 'thenlper/gte-large',
    'labeled': False,
    'chunk_size': 200,
    'chunk_overlap': 40,
    'topk': 15
}