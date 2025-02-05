from transformers import BertTokenizer, BertModel
import torch
import pandas as pd

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

chunk_file = pd.read_csv('resultados_extracao.csv')
chunks = chunk_file["Chunk"].to_list()


def tokenize_chunks(chunks, tokenizer):
    return tokenizer(chunks, return_tensors='pt', truncation=True, padding=True)
    tokenized_chunks = []

    for chunk in chunks:
        tokens = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True)
        tokenized_chunks.append(tokens)
    return tokenized_chunks

tokenized_chunks = tokenize_chunks(chunks, tokenizer)

def get_embeddings(tokens_list, model):

    with torch.no_grad():
        output =model(**tokens_list)
        last_hidden_state = output.last_hidden_state
        sentence_embedding = last_hidden_state.mean(dim=1)
    
    return sentence_embedding

bert_embeddings = get_embeddings(tokenized_chunks, model)
torch.save(bert_embeddings, "bert_embeddings.pt")
print(bert_embeddings.shape)

            