import os
import argparse
import json
import pickle
import json

import numpy as np
import transformers
import torch
from torch.nn import CrossEntropyLoss
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
from datasets import load_dataset

from cbralm.model_utils import load_model_and_tokenizer
from cbralm.file_utils import print_args
from cbralm.manage_project import create_project, add_to_project


def evaluate_logprob_with_retrieved_docs(
    model,
    tokenizer,
    searcher,
    device,
    encodings,
    begin_loc,
    end_loc,
    trg_len,
    retrieved_item,
    ranking_strategy,
    num_tokens_to_rank,
    retrieval_max_length,
    num_docs=-1,
    model_layer=None,
    *args,
):
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)

    if ranking_strategy == "first":
        assert num_docs in [
            -1,
            1,
        ], f"In 'first' ranking strategy, unexpected number of docs to rank: {num_docs}"
        num_docs = 1
        chosen_doc_id = 0
    elif ranking_strategy == "random":
        chosen_doc_id = np.random.randint(num_docs)
        retrieved_item["retrieved_docs"] = [
            retrieved_item["retrieved_docs"][chosen_doc_id]
        ]
        num_docs = 1
    elif ranking_strategy == "colbert":
        assert (
            model_layer is not None
        ), "ColBert was selected, but now model layer was specified"
        best_doc = None
        for doc_info in retrieved_item["reranked_retrieved_docs"][
            f"layer{model_layer}"
        ]:
            if best_doc is None or doc_info["rank"] > best_doc["rank"]:
                best_doc = doc_info
        chosen_doc_id = best_doc["docid"]
        num_docs = 1
        retrieved_item["retrieved_docs"] = [best_doc]
    else:
        raise NotImplementedError("Unknown Reranking Strategy")

    num_docs_in_retrieved = len(retrieved_item["retrieved_docs"])
    num_docs = (
        min(num_docs, num_docs_in_retrieved) if num_docs > 0 else num_docs_in_retrieved
    )

    input_ids = input_ids.repeat(num_docs, 1)
    target_ids = input_ids.clone()
    labels_for_ranking = input_ids.clone()
    assert input_ids.size() == (num_docs, end_loc - begin_loc)

    for doc_id in range(num_docs):
        retrieved_example = retrieved_item["retrieved_docs"][doc_id]

        doc_text = json.load(searcher.doc(retrieved_example["docid"]).raw())["contents"]

        # Changing this
        input_ids[doc_id, : len(encoded_retrieved_text)] = torch.tensor(
            encoded_retrieved_text, device=device
        )
        # to this
        # input_ids[doc_id].concat(torch.tensor(encoded_retrieved_text, device=device))

    loss_fct = CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        lm_logits = model(input_ids).logits

        # Rank:
        if ranking_strategy in ["first", "random", "colbert"]:
            batch_doc_id = 0
        else:
            if ranking_strategy == "oracle":
                labels_for_ranking[:, :-trg_len] = -100
                num_tokens_to_rank = trg_len  # We override this variable as it's not really relevant in oracle setting
            else:
                labels_for_ranking[:, : -trg_len - num_tokens_to_rank] = -100
                labels_for_ranking[:, -trg_len:] = -100
            labels_for_ranking = labels_for_ranking[:, 1:]
            assert (
                torch.sum(labels_for_ranking[0] != -100).cpu().item()
                == num_tokens_to_rank
            )

            lm_logits_for_ranking = lm_logits[..., :-1, :]
            ranking_loss = loss_fct(
                lm_logits_for_ranking.reshape(-1, lm_logits_for_ranking.size(-1)),
                labels_for_ranking.reshape(-1),
            )
            ranking_loss = ranking_loss.view(num_docs, -1)
            per_doc_ranking_loss = torch.sum(ranking_loss, dim=-1)
            chosen_doc_id = torch.argmin(per_doc_ranking_loss).cpu().item()
            batch_doc_id = chosen_doc_id

        # Calculate logprob of the chosen doc:
        lm_logits = lm_logits[batch_doc_id, -trg_len - 1 : -1, :]
        labels = target_ids[batch_doc_id, -trg_len:]  # Changed this
        loss = loss_fct(lm_logits, labels)
        token_ppls = loss.cpu()
        tokens_to_predict = labels.view(-1).cpu().tolist()
        loss = token_ppls.sum()

    return loss, chosen_doc_id, token_ppls.tolist(), tokens_to_predict


def eval_dataset(
    model,
    tokenizer,
    searcher,
    dataset,
    device,
    max_length,
    output_dir=None,
    stride=4,
    normalization_level="word",
    retrieval_info=None,
    retrieval_max_length=256,
    ranking_strategy="first",
    num_docs_to_rank=1,
    num_tokens_to_rank_logprob=16,
    model_layer=None,  # Used for ColBERT
):
    encodings = tokenizer(dataset, add_special_tokens=False, return_tensors="pt")

    print("Max context length:", max_length)
    # Number of tokens in dataset
    dataset_len = encodings.input_ids.size(1)
    print("Dataset length:", dataset_len)

    if normalization_level == "word":
        counter = dataset.count(" ")
    elif normalization_level == "token":
        counter = dataset_len
    else:
        raise ValueError(f"Unknown normalization_level: '{normalization_level}'")

    print("Normalization factor (num tokens/words..):", counter)


    # Get the retrieved dataset
    retrieval_dataset = None
    if retrieval_info:
        retrieval_dataset = retrieval_info["query_to_retrieved_docs"]

    nlls = []
    prev_end_loc = 0

    idx = 0
    all_token_ppls = []
    all_tokens_to_predict = []
    all_chosen_doc_ids = [None]
    num_inputs_no_retrieval = 0
    for begin_loc in tqdm(
        range(0, dataset_len, stride)[:100]
    ):  # Change this before benchmarking
        end_loc = min(begin_loc + max_length, dataset_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        if (
            idx > 0
            and retrieval_dataset is not None
            and len(retrieval_dataset[idx]["retrieved_docs"]) > 0
        ):
            retrieved_example = retrieval_dataset[idx]
            assert (
                retrieved_example["begin_location"] == prev_end_loc
            ), f"{retrieved_example['begin_location']} is different from {prev_end_loc}"
            assert (
                retrieved_example["end_location"] == end_loc
            ), f"{retrieved_example['end_location']} is different from {end_loc}"

            neg_log_likelihood, chosen_doc_id, token_ppls, tokens_to_predict = (
                evaluate_logprob_with_retrieved_docs(
                    model,
                    tokenizer,
                    searcher,
                    device,
                    encodings,
                    begin_loc,
                    end_loc,
                    trg_len,
                    retrieved_example,
                    ranking_strategy=ranking_strategy,
                    num_tokens_to_rank=num_tokens_to_rank_logprob,
                    retrieval_max_length=retrieval_max_length,
                    num_docs=num_docs_to_rank,
                    model_layer=model_layer,
                )
            )
            all_chosen_doc_ids.append(chosen_doc_id)
        else:
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # Calculate per-token loss
                if trg_len < max_length:
                    neg_log_likelihood = outputs.loss * trg_len
                    lm_logits = outputs.logits[..., -trg_len - 1 : -1, :]
                    labels = target_ids[..., -trg_len:]
                else:
                    neg_log_likelihood = outputs.loss * (max_length - 1)
                    lm_logits = outputs.logits[..., :-1, :]
                    labels = target_ids[..., 1:]
                neg_log_likelihood = (
                    neg_log_likelihood.to(torch.float32).squeeze().cpu()
                )
                lm_logits = lm_logits.to(torch.float32)

                loss_fct = CrossEntropyLoss(reduction="none")
                loss = loss_fct(
                    lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)
                ).cpu()
                token_ppls = loss.tolist()
                tokens_to_predict = labels.view(-1).cpu().tolist()

        nlls.append(neg_log_likelihood)
        all_token_ppls.append(token_ppls)
        all_tokens_to_predict.append(tokens_to_predict)
        assert len(all_token_ppls) == len(all_tokens_to_predict)

        prev_end_loc = end_loc
        idx += 1
        if end_loc == dataset_len:
            break

    # TODO: Add this assert back
    #assert retrieval_dataset is None or len(retrieval_dataset) == idx

    ppl = torch.exp(torch.stack(nlls).sum() / counter).item()
    print("Perplexity:", ppl)
    ppl_to_assert = np.exp(sum([sum(x) for x in all_token_ppls]) / counter)
    assert np.abs(ppl - ppl_to_assert) < 1e-3, f"{ppl:.3f}, {ppl_to_assert:.3f}"

    if output_dir is not None:
        d = {"eval_perplexity": ppl}
        if retrieval_dataset is not None:
            d["num_input_no_retrieval"] = num_inputs_no_retrieval
        with open(os.path.join(output_dir, "eval.json"), "w") as f:
            f.write(json.dumps(d) + "\n")

        with open(os.path.join(output_dir, "ppls.pkl"), "wb") as f:
            to_dump = (
                (all_token_ppls, all_tokens_to_predict, all_chosen_doc_ids)
                if all_chosen_doc_ids
                else (all_token_ppls, all_tokens_to_predict)
            )
            pickle.dump(to_dump, f)


def main(args):

    model, tokenizer, config, device = load_model_and_tokenizer(
        args.model_name,
        model_parallelism=args.model_parallelism,
        cache_dir=args.cache_dir,
        auth_token=args.auth_token,
    )

    # Model context size (e.g., 1024 for GPT-2)
    max_length = args.max_length
    model_max_length = (
        config.n_positions
        if hasattr(config, "n_positions")
        else config.max_position_embeddings
    )
    if max_length is None or max_length > model_max_length:
        max_length = model_max_length

    if args.load_from == "hf":
        dataset = load_dataset(
            args.dataset_path, args.dataset_name, split=args.dataset_split
        )
        dataset = "".join([x["text"] if x["text"] else " \n" for x in dataset])
    else:
        with open(args.dataset_path, "r") as f:
            dataset = f.read()

    transformers.logging.set_verbosity_error()
    retrieval_info = None 
    if args.retrieved_file is not None:
        with open(args.retrieved_file, "r") as f:
            retrieval_info = json.load(f)

    if retrieval_info is not None and not os.path.isdir(args.project_name):
        raise FileNotFoundError(f"Project {args.project_name} doesn't exist.")

    save_dir = args.project_name
    if retrieval_info is None:
        save_dir = create_project(name=args.project_name)

    benchmark_dir = os.path.join(save_dir, "benchmark")
    os.makedirs(benchmark_dir, exist_ok=True)

    output_dir = add_to_project(module=args.run_name, parent=benchmark_dir)

    print_args(args, output_dir=output_dir, retrieval_info=retrieval_info)

    # Create Searcher
    searcher = LuceneSearcher.from_prebuilt_index(args.retrieval_corpus)

    eval_dataset(
        model,
        tokenizer,
        searcher,
        dataset,
        device,
        max_length=max_length,
        output_dir=output_dir,
        stride=args.stride,
        normalization_level=args.normalization_level,
        retrieval_info=retrieval_info,
        retrieval_max_length=args.retrieved_max_length,
        ranking_strategy=args.ranking_strategy,
        num_docs_to_rank=args.num_docs_to_rank,
        num_tokens_to_rank_logprob=args.ranking_logprob_past_tokens,
        model_layer=args.model_layer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str)
    parser.add_argument("--project-name", type=str)

    # Model params
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--model-parallelism", action="store_true")
    parser.add_argument("--auth-token", type=str, default=None)

    # Dataset params
    parser.add_argument("--load-from", type=str, choices=["hf", "file"], default="hf")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--dataset-split", type=str, default="test")
    parser.add_argument(
        "--normalization_level", choices=["word", "token"], default="word"
    )
    parser.add_argument(
        "--retrieval-corpus", type=str, default=None
    )  # wikipedia-dpr-100w

    # retrieval params
    parser.add_argument("--retrieved-file", type=str, default=None)
    parser.add_argument("--retrieved-max-length", type=int, default=256)
    parser.add_argument(
        "--ranking-strategy",
        type=str,
        choices=["first", "logprob", "oracle", "random", "colbert"],
        default="first",
    )
    parser.add_argument("--num-docs-to-rank", type=int, default=-1)
    parser.add_argument("--ranking-logprob-past-tokens", type=int, default=16)

    # ColBERT params
    parser.add_argument(
        "--model-layer",
        type=int,
        default=None,
        help="Which layer to use from the reranker",
    )

    args = parser.parse_args()

    main(args)
