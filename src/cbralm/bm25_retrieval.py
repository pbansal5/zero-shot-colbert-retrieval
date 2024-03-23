import datasets
from pathlib import Path, PurePath
from pyserini.search.lucene import LuceneSearcher
import argparse
from transformers import AutoTokenizer
import json
import tqdm
import os
import logging
import multiprocessing

from .manage_project import create_project
from .file_utils import print_args


def _not_forbidden(context, forbidden_titles):
    title, _ = context.split("\n")
    if title.startswith('"') and title.endswith('"'):
        title = title[1:-1]
    return title not in forbidden_titles


def get_bm25_documents(args):
    output_json = {}
    query_to_retrieved_docs = []

    if args.forbidden_titles:
        with open(args.forbidden_titles, "r") as f:
            forbidden_titles = [line.strip() for line in f]
        forbidden_titles = set(forbidden_titles)
    else:
        forbidden_titles = set([])

    if args.data_dir:
        datasets.config.DOWNLOADED_DATASETS_PATH = Path(args.data_dir)
        datasets.config.HF_DATASETS_CACHE = Path(args.data_dir)

    # match args.query_corpus:
    #     case 'wikitext':
    #         query_corpus = datasets.load_dataset('wikitext','wikitext-103-v1')['test']['text']
    #     case default:
    #         raise Exception("Unknown Query Corpus")

    if args.query_corpus == "wikitext":
        query_corpus = datasets.load_dataset("wikitext", "wikitext-103-v1")["test"][
            "text"
        ]
    else:
        raise Exception("Unknown Query Corpus")

    query_corpus = " ".join(query_corpus).strip()
    searcher = LuceneSearcher.from_prebuilt_index(args.retrieval_corpus)

    ## We tokenize the query corpus and untokenize it since the stride
    ## is in terms of number of tokens instead of number of words

    num_docs_to_retrieve = max(4 * args.topK, 100)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenized_query_corpus = tokenizer(query_corpus, return_tensors="np")["input_ids"][
        0
    ]
    length_query_corpus = len(tokenized_query_corpus)

    prev_end_location = 0

    logging.info("Creating Queries...")
    for start in tqdm.tqdm(
        range(0, length_query_corpus, args.retrieval_stride)
    ):  # Change this after debugging
        end = min(start + args.max_length, length_query_corpus)
        target_beg_location = prev_end_location

        query_to_retrieved_docs.append(
            {
                "begin_location": target_beg_location,
                "end_location": end,
                "future": tokenizer.decode(
                    tokenized_query_corpus[target_beg_location:end]
                ),
            }
        )
        prev_end_location = end

    logging.info("Querying Retriever...")
    batch_size = 1000

    def get_query_string(start_loc):
        prefix_tokens = tokenized_query_corpus[:start_loc]
        query_tokens = prefix_tokens[-args.retrieval_query_length :]
        query_str = tokenizer.decode(query_tokens)
        return query_str

    for i in tqdm.tqdm(range(0, len(query_to_retrieved_docs), batch_size)):
        query_data = query_to_retrieved_docs[i:i+batch_size]

        query_string = [get_query_string(d["begin_location"]) for d in query_data]

        assert len(query_string) == len(query_data)

        all_res = searcher.batch_search(
            query_string,
            qids=[str(i) for i in range(len(query_string))],
            k=num_docs_to_retrieve,
            threads=multiprocessing.cpu_count(),
        )

        for qid, res in all_res.items():
            qid = int(qid)
            d = query_data[qid]
            d["query"] = query_string[qid]
            filtered_hits = [
                hit
                for hit in res
                if _not_forbidden(json.loads(hit.raw)["contents"], forbidden_titles)
            ]

            filtered_hits_topk = [
                dict(
                    {
                        "rank": i,
                        "docid": filtered_hits[i].docid,
                        "score": filtered_hits[i].score,
                        "text": json.loads(filtered_hits[i].raw)["contents"],
                    }
                )
                for i in range(min(args.topK, len(filtered_hits)))
            ]
            d["retrieved_docs"] = filtered_hits_topk

    output_json["query_to_retrieved_docs"] = query_to_retrieved_docs
    output_json["bm25_logging_info"] = vars(args)
    return output_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset Information
    parser.add_argument(
        "--data-dir", type=str, default=None
    )  # '/datastor1/pbansal/huggingface_cache', /home/pb25659/huggingface_cache
    parser.add_argument(
        "--retrieval-corpus", type=str, default=None
    )  # wikipedia-dpr-100w
    parser.add_argument("--query-corpus", type=str, default=None)  # wikipedia-dpr-100w
    parser.add_argument(
        "--forbidden-titles", type=str, default=None
    )  # jsons/wikitext_forbidden_titles.txt

    # Project Info
    parser.add_argument("--project-name", type=str, default=None)  # Name of the Run

    # Retrieval Info
    parser.add_argument("--topK", type=int, default=16)  # 16
    parser.add_argument("--retrieval-type", type=str, default=None)

    # Model Info
    parser.add_argument("--retrieval-query-length", type=int, default=32)  # 32
    parser.add_argument("--retrieval-stride", type=int, default=4)  # 4
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--tokenizer", type=str, default=None)  # gpt2

    args = parser.parse_args()

    # Create the project run
    logging.info("Creating Project...")
    save_path = create_project(args.project_name)

    # Log Information
    print_args(args, output_dir=save_path)

    # Run Retrieval
    output_json = get_bm25_documents(args)

    # Save Dependencies
    with open(os.path.join(save_path, "run.json"), "w") as f:
        json.dump(output_json, f)
