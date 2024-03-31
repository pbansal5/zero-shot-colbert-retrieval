import json
import sys
import argparse

from tqdm import tqdm

from ralm.file_utils import print_args
from ralm.model_utils import load_model_and_tokenizer
from ralm.rerankers.reranker_factory import add_reranker_args, get_reranker 

RERANKING_TYPES = [
    "colbert"
]


def main(args):
    # Dump args
    print_args(args, output_file=args.output_file.replace(".json", ".args.txt"))

    with open(args.retrieved_file, "r") as f:
        retrieval_dataset = json.load(f)
    print("Queries to process:", len(retrieval_dataset))

    model, tokenizer, config, device = load_model_and_tokenizer(
        args.model_name, model_parallelism=args.model_parallelism, cache_dir=args.cache_dir, auth_token=args.auth_token
    )

    print(f"Creating reranker of type {args.reranking_type}...")
    reranker = get_reranker(args.reranking_type, args, tokenizer, model, device)


    print("Reranking Documents...")
    for query_index in tqdm(range(0,len(retrieval_dataset),args.batch_size)):
        query_info = retrieval_dataset[query_index:query_index+args.batch_size]
        reranker.rerank(query_info, k=args.num_docs_to_rank)

    print(f"Finished processing {len(retrieval_dataset)} queries")
    print(f"Writing to {args.output_file}")
    with open(args.output_file, "w") as f:
        f.write(json.dumps(retrieval_dataset, indent=4))
        f.write("\n")

    print("Done!")


if __name__ == '__main__':
    assert sys.argv[1] == "--reranking_type"
    retrieval_type = sys.argv[2]

    assert retrieval_type in RERANKING_TYPES 

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_file", required=True, type=str)

    # Model params
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model_parallelism", action="store_true")
    parser.add_argument("--auth_token", type=str, default=None)

    # Reranking params
    parser.add_argument("--retrieved_file", type=str, required=True)
    parser.add_argument("--num_docs_to_rank", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--reranking_type", type=str, required=True, choices=RERANKING_TYPES, default="colbert")
    add_reranker_args(parser, retrieval_type)

    args = parser.parse_args()
    main(args)

