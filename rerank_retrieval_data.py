import json
import sys
import argparse
import logging

from tqdm import tqdm

from ralm.file_utils import print_args
from ralm.model_utils import load_model_and_tokenizer
from ralm.rerankers.reranker_factory import add_reranker_args, get_reranker 

RERANKING_TYPES = [
    "zs-llms", # Zero Shot Contrastive Late Interaction Language Models
    "finegrain", # Finegrain Inter-layer embedding Reranking
    "coarse", # Coarse Reranking
    "contriever", # https://arxiv.org/pdf/2112.09118.pdf
    "spider", # https://arxiv.org/pdf/2112.07708.pdf
    "dpr" # https://arxiv.org/pdf/2004.04906.pdf
]


def main(args):
    # Dump args
    print_args(args, output_file=args.output_file.replace(".json", ".args.txt"))

    with open(args.retrieved_file, "r") as f:
        retrieval_dataset = json.load(f)
    logging.info(f"Queries to process: {len(retrieval_dataset)}")

    model, tokenizer, config, device = load_model_and_tokenizer(
        args.model_name,
        model_parallelism=args.model_parallelism,
        cache_dir=args.cache_dir,
        auth_token=args.auth_token,
        model_type = args.model_type
    )

    logging.info(f"Creating reranker of type {args.reranking_type}...")
    reranker = get_reranker(args.reranking_type, args, tokenizer, model, device)


    logging.info("Reranking Documents...")
    final_document = len(retrieval_dataset) if args.num_queries_to_test is None else args.num_queries_to_test
    for query_index in tqdm(range(1, final_document, args.batch_size)):
        query_info = retrieval_dataset[query_index:min(query_index+args.batch_size, final_document)]
        reranker.rerank(query_info, k=args.num_docs_to_rank)

    logging.info(f"Finished processing {len(retrieval_dataset)} queries")
    logging.info(f"Writing to {args.output_file}")
    with open(args.output_file, "w") as f:
        f.write(json.dumps(retrieval_dataset, indent=4))
        f.write("\n")

    logging.info("Done!")


if __name__ == '__main__':
    assert sys.argv[1] == "--reranking_type"
    retrieval_type = sys.argv[2]

    assert retrieval_type in RERANKING_TYPES 

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_file", required=True, type=str)

    # Model params
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--model_type", type=str, default="causal")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model_parallelism", action="store_true")
    parser.add_argument("--auth_token", type=str, default=None)

    # Reranking params
    parser.add_argument("--retrieved_file", type=str, required=True)
    parser.add_argument("--num_docs_to_rank", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--reranking_type", type=str, required=True, choices=RERANKING_TYPES, default="colbert")
    add_reranker_args(parser, retrieval_type)

    # Testing params
    parser.add_argument("--num_queries_to_test", default=None, type=int)

    args = parser.parse_args()
    main(args)

