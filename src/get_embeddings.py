import torch
import numpy as np
import datasets
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer, GPT2Tokenizer, GPT2Model
import transformers
import os 
import tqdm

data_path = '/datastor1/pbansal/huggingface_cache'
embedding_out_path = '%s/wiki_dpr_embeddings'%data_path
model_to_use = 'bert-base-uncased' # 'roberta', 'gpt2'

datasets.config.DOWNLOADED_DATASETS_PATH = Path(data_path)
datasets.config.HF_DATASETS_CACHE = Path(data_path)

wiki_retrieval = datasets.load_dataset('wiki_dpr','psgs_w100.multiset.no_index.no_embeddings')['train']


if (model_to_use.split('-')[0] == 'bert'):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased',cache_dir=data_path)
    if (model_to_use.split('-')[1] == 'base'):
        num_layers = 13
    elif (model_to_use.split('-')[1] == 'large'):
        num_layers = 25
    else : 
        Exception("use either bert-base or bert-large")
    
elif (model_to_use.split('-')[0] == 'roberta'):
    tokenizer = AutoTokenizer.from_pretrained('roberta-base',cache_dir=data_path)
    if (model_to_use.split('-')[1] == 'base'):
        num_layers = 13
    elif (model_to_use.split('-')[1] == 'large'):
        num_layers = 25
    else : 
        Exception("use either roberta-base or roberta-large")

elif (model_to_use.split('-')[0] == 'gpt2'):
    tokenizer = AutoTokenizer.from_pretrained('gpt2',cache_dir=data_path)
else : 
    Exception("use valid model_name")

model = AutoModel.from_pretrained(model_to_use,cache_dir=data_path)

batch_size = 500
max_length = 256
embedding_size = 10
batch_indices_list = [np.arange(i,np.minimum(batch_size+i,len(wiki_retrieval))) for i in range(0,len(wiki_retrieval),batch_size)]
representations = [torch.zeros((len(wiki_retrieval),max_length,embedding_size)).float() for _ in range(num_layers)]

for batch_indices in tqdm.tqdm(batch_indices_list):
    paragraphs = [wiki_retrieval[j]['text'] for j in batch_indices]
    tokenized_inputs = tokenizer(paragraphs,truncation=True,max_length=max_length,padding='max_length',return_tensors='pt')['input_ids']
    model_hidden_states = model(tokenized_inputs,output_hidden_states=True)['hidden_states']
    assert num_layers == len(model_hidden_states)
    for layer_ in range(num_layers):
        representations[layer_][batch_indices] = model_hidden_states[layer_]

for layer_ in tqdm.tqdm(num_layers): 
    torch.save(representations[layer_],'%s/%s_layer-%d.pt'%(embedding_out_path,model_to_use,layer_))
