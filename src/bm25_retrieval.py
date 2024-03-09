import datasets
from pathlib import Path,PurePath
from pyserini.search.lucene import LuceneSearcher
import argparse
from transformers import AutoTokenizer
import json
import tqdm
import os

def not_forbidden(context,forbidden_titles):
    title, _ = context.split("\n")
    if title.startswith('"') and title.endswith('"'):
        title = title[1:-1]
    return (title not in forbidden_titles)

def get_bm25_documents(args):
    output_json = vars(args)
    query_to_retrieved_docs = dict({})


    if (args.forbidden_titles != ''):
        with open(args.forbidden_titles, "r") as f:
            forbidden_titles = [line.strip() for line in f]
        forbidden_titles = set(forbidden_titles)
    else : 
        forbidden_titles = set([])

    if (args.data_dir != ''):
        datasets.config.DOWNLOADED_DATASETS_PATH = Path(args.data_dir)
        datasets.config.HF_DATASETS_CACHE = Path(args.data_dir)

    if (args.query_corpus == 'wikitext'):
        query_corpus = datasets.load_dataset('wikitext','wikitext-103-v1')['test']['text']
    else : 
        raise Exception("Unknown Query Corpus")
    
    query_corpus = ' '.join(query_corpus).strip()
    searcher = LuceneSearcher.from_prebuilt_index(args.retrieval_corpus)

    ## We tokenize the query corpus and untokenize it since the stride is in terms of number of tokens instead of number of words

    num_docs_to_retrieve = max(4*args.topK,100)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenized_query_corpus = tokenizer(query_corpus,return_tensors='np')['input_ids'][0]
    length_query_corpus = len(tokenized_query_corpus)

    for start_ in tqdm.tqdm(range(0,length_query_corpus,args.retrieval_stride)):
        tokenized_query_segment = tokenized_query_corpus[start_:start_+args.retrieval_query_length]
        query_segment = tokenizer.decode(tokenized_query_segment)
        hits = searcher.search(query_segment,num_docs_to_retrieve)
        
        # some paragraphs are shared in the retrieval corpus and wikitext-103. Hence these paragraphs need to be removed from the retrieval corpus
        # This follows in-context RALM paper. 
        filtered_hits = [hit for hit in hits if not_forbidden(json.loads(hit.raw)['contents'],forbidden_titles)]

        filtered_hits_topk = [dict({'rank':i,'docid':filtered_hits[i].docid,'score':filtered_hits[i].score}) for i in range(args.topK)]
        query_to_retrieved_docs[query_segment] = filtered_hits_topk

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
    parser.add_argument('--forbidden_titles', type=str) # jsons/wikitext_forbidden_titles.txt
    args = parser.parse_args()

    output_json = get_bm25_documents(args)

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_file, 'w') as f:
        json.dump(output_json, f)


