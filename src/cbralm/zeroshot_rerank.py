import torch
import numpy as np
import datasets
from pathlib import Path
from pyserini.search.lucene import LuceneSearcher
from transformers import AutoModel, AutoTokenizer
import transformers
import os
import tqdm
import argparse
import json
import logging

from manage_project import add_to_project
from file_utils import print_args


def MaxSim(query_embed, docs_embed):
    # query_embed has shape 1 x (# of query tokens) x (rep_dim)
    # docs_embed has shape (# of docs) x (# of query tokens) x (rep_dim)

    query_embed_normalized = (
        query_embed / torch.norm(query_embed, dim=-1).clamp(min=1e-5)[:, :, None]
    )
    docs_embed_normalized = (
        docs_embed / torch.norm(docs_embed, dim=-1).clamp(min=1e-5)[:, :, None]
    )

    tokenwise_similarity = (
        query_embed_normalized[:, :, None, :] * docs_embed_normalized[:, None, :, :]
    ).sum(axis=-1)
    # shape of tokenwise_similarity is (# of docs) x (# of query tokens) x (# of document tokens).
    # as an example, if there are 16 retrieved documents for each query, and there are 256 tokens in both query and document,
    # tokenwise_similarity is 16x256x256

    max_over_doctokens_similarity = torch.max(
        tokenwise_similarity, dim=2
    ).values  # max_over_doctokens_similarity has shape (# of docs) x (# of query tokens)
    sum_over_querytokens_similarity = max_over_doctokens_similarity.sum(
        dim=1
    )  # sum_over_querytokens_similarity has shape (# of docs)

    nonzero_tokens_query = torch.norm(query_embed_normalized, dim=-1).sum(dim=1)[
        0
    ]  # query_embed_normalized has norm 1 if token is nonzer and 0 if token is zero.
    # hence summing over the norms of all query tokens (i.e. over 1) gives the number of non-zero tokens

    avg_over_querytokens_similarity = (
        sum_over_querytokens_similarity / nonzero_tokens_query
    )
    order = torch.argsort(sum_over_querytokens_similarity, descending=True)

    return order, avg_over_querytokens_similarity[order].cpu().tolist()


@torch.no_grad()
def zeroshot_rerank(args):

    with open(args.bm25_file, "r") as f:
        doc_retrieval = json.load(f)
        query_to_retrieved_docs = doc_retrieval["query_to_retrieved_docs"]

    logging.info("Creating Model...")
    if args.data_dir:
        datasets.config.DOWNLOADED_DATASETS_PATH = Path(args.data_dir)
        datasets.config.HF_DATASETS_CACHE = Path(args.data_dir)

    if args.rerank_model.split("-")[0] == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif args.rerank_model.split("-")[0] == "roberta":
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    elif args.rerank_model.split("-")[0] == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise Exception("Use valid rerank_model")

    model = AutoModel.from_pretrained(args.rerank_model).cuda()
    searcher = LuceneSearcher.from_prebuilt_index(args.retrieval_corpus)

    logging.info("ReRanking Queries")
    for query_info in tqdm.tqdm(query_to_retrieved_docs):
        query = query_info["query"]
        retrieved_docs = query_info["retrieved_docs"]

        # Reranking with the ColBert objective
        sentences = [query] + [
            json.loads(searcher.doc(doc["docid"]).raw())["contents"]
            for doc in retrieved_docs
        ]
        tokenized_inputs = tokenizer(
            sentences,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        model_hidden_states = model(
            tokenized_inputs["input_ids"].cuda(),
            attention_mask=tokenized_inputs["attention_mask"].cuda(),
            output_hidden_states=True,
        )["hidden_states"]
        model_hidden_states = [
            hidden_state * tokenized_inputs["attention_mask"][:, :, None].cuda()
            for hidden_state in model_hidden_states
        ]

        reranked_layerwise_docs = dict({})
        for layer, hidden_state in enumerate(model_hidden_states):
            order, scores = MaxSim(hidden_state[0:1], hidden_state[1:])
            reranked_layerwise_docs[f"layer{layer}"] = [
                dict(
                    {
                        "rank": i,
                        "docid": retrieved_docs[order[i]]["docid"],
                        "score": scores[i],
                        "text": retrieved_docs[order[i]]["text"],
                    }
                )
                for i in range(len(retrieved_docs))
            ]

        query_info["reranked_retrieved_docs"] = reranked_layerwise_docs

    doc_retrieval["reranking_args"] = vars(args)

    return doc_retrieval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default=None
    )  # '/datastor1/pbansal/huggingface_cache', /home/pb25659/huggingface_cache
    parser.add_argument(
        "--retrieval_corpus", type=str, default=None
    )  # wikipedia-dpr-100w

    parser.add_argument("--project-name", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--rerank_model", type=str, default=None)
    parser.add_argument("--topK", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    # Create project submodule
    if not os.path.isdir(args.project_name):
        assert FileNotFoundError(f"Project {args.project_name} does not exits")
    save_path = add_to_project(module=args.run_name, parent=args.project_name)

    # Log Information
    print_args(args, output_dir=save_path)

    # Run Retrieval
    output_json = zeroshot_rerank(args)

    logging.info("Saving Run...")
    with open(os.path.join(save_path, "run.json"), "w") as f:
        json.dump(output_json, f)
