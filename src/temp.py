import datasets
from pathlib import Path
from pyserini.search.lucene import LuceneSearcher

data_path = '/datastor1/pbansal/huggingface_cache'

datasets.config.DOWNLOADED_DATASETS_PATH = Path(data_path)
datasets.config.HF_DATASETS_CACHE = Path(data_path)

wiki_retrieval = datasets.load_dataset('wiki_dpr','psgs_w100.multiset.no_index.no_embeddings')['train']

searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr-100w')
hits = searcher.search('what is a lobster roll?')

for i in range(0, 10):
    print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')
    print (wiki_retrieval[hits[i].docid]['text'])
