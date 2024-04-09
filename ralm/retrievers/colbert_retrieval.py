import json
import logging

import colbert
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection 

from ralm.retrievers.base_retrieval import BaseRetriever




class ColbertRetriever(BaseRetriever):
    def __init__(self, tokenizer, index_name, num_tokens_for_query, forbidden_titles_path):
        super(SparseRetriever, self).__init__(tokenizer=tokenizer)
        indexer, dataset = self._get_indexer_and_collection(index_name)
        self.searcher = self._get_searcher(index_name, collection)
        self.dataset = dataset 
        self.num_tokens_for_query = num_tokens_for_query

        self.forbidden_titles = self._get_forbidden_titles(forbidden_titles_path)

    def _get_indexer_and_collection(
            self,
            index_name: str,
            create_index: str = False,
            model_checkpoint: str = None,
            dataset_name: str = None,
            dataset_configuration: str = None,
            dataset_split: str = None,
            nranks: int = 1,
            n_bits: int = 16,
            kmeans_niters : int = 4,
            doc_maxlength: int = 300,
        ):
        if not create_index:
            try:
                logging.info(f"Attempting to download the index from Huggingface")
                from huggingface_hub import snapshot_download
                return snapshot_download(repo_id=index_name)

            except ValueError:
                logging.info(f"Index does not exist on Huggingface")
                logging.info("Attempting to treat the index as a directory.")
                return index_name

        from datasets import load_dataset
        logging.info("Attempting to download dataset from Huggingface")
        dataset = load_datset(dataset_name, dataset_configuration, split=dataset_split)

        logging.info(f"Processing text from {dataset_name} - {dataset_configuration} - {dataset_split}")
        collection = [x["text"] for x in dataset]

        index_name = f'{dataset_name}.{dataset_configuration}.{dataset_split}.{n_bits}'

        with Run().context(RunConfig(nranks=nranks, experiment="ralm")):
            config = ColBERTConfig(doc_maxlen=doc_maxlength, nbits=n_bits, kmeans_niters=kmeans_niters)

            indexer = Indexer(checkpoint=model_checkpoint, config=config)
            indexer.index(name=index_name, collection=collection, overwrite=True)

        return index_name, collection, dataset

    def _get_searcher(index_name, collection):
        with Run().context(RunConfig(experiment="ralm")):
            return Searcher(index=index_name, collection=collection)


    def _get_forbidden_titles(self, forbidden_titles_path):
        if forbidden_titles_path is None:
            return []
        with open(forbidden_titles_path, "r") as f:
            forbidden_titles = [line.strip() for line in f]
        return set(forbidden_titles)

    def _get_title_from_retrieved_document(self, doc):
        title, _ = doc.split("\n")
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        return title

    def _retrieve_no_forbidden(self, query_str):
        k = 16
        prev_k = 0
        while True:
            retrieved_res = self.searcher.search(query_str, k=k)
            for idx in range(prev_k, k):
                res_dict = json.loads(retrieved_res[idx].raw)
                context_str = res_dict["contents"]
                title = self._get_title_from_retrieved_document(context_str)
                if title not in self.forbidden_titles:
                    return context_str
            prev_k = k
            k *= 2

    def _get_query_string(self, sequence_input_ids, target_begin_location, target_end_location, title=None):
        # We isolate the prefix to make sure that we don't take tokens from the future:
        prefix_tokens = sequence_input_ids[0, :target_begin_location]
        query_tokens = prefix_tokens[-self.num_tokens_for_query:]
        query_str = self.tokenizer.decode(query_tokens)
        return query_str

    def retrieve(self, sequence_input_ids, dataset, k=1):
        queries = [
            self._get_query_string(
                sequence_input_ids,
                d["begin_location"],
                d["end_location"],
                d["title"] if "title" in d else None
            )
            for d in dataset
        ]
        assert len(queries) == len(dataset)

        raise NotImplementedError

        return dataset
