import json
import pickle
import argparse
import logging

from ralm.file_utils import print_args
from ralm.model_utils import load_model_and_tokenizer
from prefix_trainer.trainer import EmbeddingTrainer
from prefix_trainer.datasets import PerplexityDataset

def main(args):
    # Dump args
    print_args(args, output_file=args.output_file.replace(".json", ".args.txt"))

    objects = []
    with open(args.retrieved_file, "r") as f:
        retrieval_dataset = json.load(f)
    with open(args.score_file, "rb") as f:
        while True:
            try:
                objects.append(pickle.load(f))
            except EOFError:
                break
    assert len(objects) == 3, "Something is not right"
    best_doc_id = objects[-1]
    with open(args.training_config, "r") as f:
        training_config = json.load(f)

    train_dataset = PerplexityDataset(retrieval_info=retrieval_dataset, best_doc=best_doc_id)

    model, tokenizer, config, device = load_model_and_tokenizer(
        args.model_name,
        model_parallelism=args.model_parallelism,
        cache_dir=args.cache_dir,
        auth_token=args.auth_token,
        model_type = args.model_type
    )

    if args.embedding:
        logging.info("Creating Embedding Trainer...")
        trainer = EmbeddingTrainer(model, tokenizer, train_dataset, None, args.num_tokens, args.max_length, args.split_tower)
    else:
        logging.info("Creating Token Trainer...")
        raise NotImplementedError


    logging.info("Starting Training...")
    query_prefix, document_prefix = trainer.train(training_config, device)

    logging.info(f"Writing to {args.output_file}")
    with open(args.output_file, "wb") as f:
        pickle.dump((query_prefix, document_prefix), f)

    logging.info("Done!")


if __name__ == '__main__':
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
    parser.add_argument("--score_file", type=str, required=True)
    parser.add_argument("--training_config", type=str, required=True)
    parser.add_argument("--embedding", action="store_true", defualt=False)
    parser.add_argument("--num_tokens", type=int, default=12)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--split_tower", action="store_true", defualt=False)

    args = parser.parse_args()
    main(args)

