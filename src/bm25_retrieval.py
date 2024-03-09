import datasets
from pathlib import Path,PurePath
from pyserini.search.lucene import LuceneSearcher
import argparse
from transformers import AutoTokenizer
import json
import tqdm
import os

def get_bm25_documents(args):
    output_json = vars(args)
    query_to_retrieved_docs = dict({})

    if (args.data_dir != ''):
        datasets.config.DOWNLOADED_DATASETS_PATH = Path(args.data_dir)
        datasets.config.HF_DATASETS_CACHE = Path(args.data_dir)

    if (args.query_corpus == 'wikitext'):
        query_corpus = datasets.load_dataset('wikitext','wikitext-103-v1')['test']['text']
    else : 
        print ("Unknown Query Corpus")
        exit()
    
    query_corpus = ' '.join(query_corpus)[:1000]
    searcher = LuceneSearcher.from_prebuilt_index(args.retrieval_corpus)

    ## We tokenize the query corpus and untokenize it since the stride is in terms of number of tokens instead of number of words

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenized_query_corpus = tokenizer(query_corpus,return_tensors='np')['input_ids'][0]
    length_query_corpus = len(tokenized_query_corpus)

    for start_ in tqdm.tqdm(range(0,length_query_corpus,args.retrieval_stride)):
        tokenized_query_segment = tokenized_query_corpus[start_:start_+args.retrieval_query_length]
        query_segment = tokenizer.decode(tokenized_query_segment)
        hits = searcher.search(query_segment,args.topK)
        hits = [dict({'rank':i,'docid':hits[i].docid,'score':hits[i].score}) for i in range(args.topK)]
        query_to_retrieved_docs[query_segment] = hits

    output_json['query_to_retrieved_docs'] = query_to_retrieved_docs
    return output_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str) # '/datastor1/pbansal/huggingface_cache', /home/pb25659/huggingface_cache
    parser.add_argument('--output_file', type=str) # 
    parser.add_argument('--retrieval_corpus', type=str) # wikipedia-dpr-100w
    parser.add_argument('--query_corpus', type=str) # wikipedia-dpr-100w
    parser.add_argument('--topK', type=int) # 16
    parser.add_argument('--retrieval_query_length', type=int) # 32
    parser.add_argument('--retrieval_stride', type=int) # 4
    parser.add_argument('--tokenizer', type=str) # gpt2
    args = parser.parse_args()

    output_json = get_bm25_documents(args)

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_file, 'w') as f:
        json.dump(output_json, f)


