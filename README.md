# ZeroShot In-Context Retrieval-Augmented Language Models

## Table of Contents
- [Setup](#setup)
- [TODO](#todo)
- [Retrieval](#retrieval)
- [Reranking](#reranking)
- [Evaluation](#evaluation)
  - [Language Models](#list-of-language-models)
  - [Evaluate models w/o retrieval](#evaluate-models-wo-retrieval)
  - [Evaluate models with retrieval](#evaluate-models-with-retrieval)
  - [Evaluate models with reranking](#reranking)
- [Display Results](#display-results)
- [Acknowledgements](#acknowledgements)

## Setup

Install python 3.8 first.

To install the required libraries in our repo, run:
```bash
pip install -r requirements.txt
```
To have a Pytorch version specific to your CUDA, [install](https://pytorch.org/) your version before running the above command.

To install Java using conda, run:

```bash
conda install openjdk=11
```

## TODO 

Checkout the todo section in the [doc](https://docs.google.com/document/d/172iMLu2UBYaihZPa6I8a2wipJBKKXPkITkuS7V9YhDc/edit?usp=sharing).

## Retrieval

Currently only BM25 retrieval is supported, however this was shown to be the best in [In Context RALM](https://github.com/AI21Labs/in-context-ralm).

### BM25

Java 11 is required to run bm25 retrieval with pyserini.

```bash
python prepare_retrieval_data.py \
  --retrieval_type sparse \
  --tokenizer_name $MODEL_NAME \
  --max_length 1024 \
  --dataset_path wikitext \
  --dataset_name wikitext-103-v1 \
  --dataset_split [validation, test] \
  --index_name wikipedia-dpr \
  --forbidden_titles_path ralm/retrievers/wikitext103_forbidden_titles.txt \
  --stride 4 \
  --output_file $RETRIEVAL_FILE \
  --num_tokens_for_query 32 \
  --num_docs 16 
```

## Reranking

All reranking requires a retrieval file. First run retrieval and then run reranking.

### List of Reranking Methods


* ZS-Col-LLMs: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `bert-base-uncased`, 
* ColBERT: ` `,
* Bert: ` `,
* Contriever: ` `,

### Zero Shot Col-LLMs

Zero Shot Col-LLMs utilizes any pretrained LM as a reranker and calculates the query document similarity
using the maxsim operator. Zero Shot Col-LLMs can use either the inter-layer embeddings or the attention
projection embeddings to rerank documents.

To run reranking with zero shot Col-LLMs.

```bash
python rerank_retrieval_data.py \
  --reranking_type zs-colllms \
  --model_name gpt2 \
  --batch_size 1 \
  --output_file $OUTPUT_DIR \
  --retrieved_file $RETRIEVAL_FILE \
  --max_length 256 \
  --num_docs_to_rank 16 
```

Rather than using embeddings dimension we can also use attention matrix to project embeddings into subspaces
which can then we used to rerank the documents. To do this we can pass the `--attention` flag.

```bash
python rerank_retrieval_data.py \
  --reranking_type zs-colllms \
  --attention \
  --model_name gpt2 \
  --batch_size 1 \
  --output_file $OUTPUT_DIR \
  --retrieved_file $RETRIEVAL_FILE \
  --max_length 256 \
  --num_docs_to_rank 16 
```

### ColBERT

We can also use an OOD trained ColBERT style reranker trained on MS MARCO. While Zero Shot ColBERT could use
any model, they method specifically relies on LMs pretrained on the ColBERT objective. These towers are BERT Based.

To run reranking with the ColBert objective:

```bash

```

### BERT

Bert reranking use's the more common coarse grain reranking of a pretrained Bert Tower.

To run reranking with the BERT objective:

```bash

```

### Contriever 

Contriever is another coarse grain reranking method introduced in this [paper]().

To run reranking with a Contriever Model:

```bash

```

## Evaluation

### List of Language Models

* GPT-2: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
* GPT-Neo: `EleutherAI/gpt-neo-1.3B`, `EleutherAI/gpt-neo-2.7B`, `EleutherAI/gpt-j-6B`
* OPT: `facebook/opt-125m`, `facebook/opt-350m`, `facebook/opt-1.3b`, `facebook/opt-2.7b`, `facebook/opt-6.7b`, `facebook/opt-13b`, `facebook/opt-30b`, `facebook/opt-66b`

### Evaluate models w/o retrieval

To run evaluation on models without retrieval, please use the following command (you can increase `stride` to 32 for faster evaluation):
```bash
python eval_lm.py \
  --model_name $MODEL_NAME \
  --dataset_path wikitext \
  --dataset_name wikitext-103-v1 \
  --dataset_split [validation, test] \
  --output_dir $OUTPUT_DIR \
  --stride 4 \
  --max_length 1024 \
  [--model_parallelism]
```

### Evaluate models with retrieval:

To run models with retrieval, use the `$RETRIEVAL_FILE` output from the `prepare_retrieval_data.py` script:
```bash
python eval_lm.py \
  --model_name $MODEL_NAME \
  --dataset_path wikitext \
  --dataset_name wikitext-103-v1 \
  --dataset_split [validation, test] \
  --output_dir $OUTPUT_DIR \
  --stride 4 \
  --max_length 1024 \
  [--model_parallelism] \
  --retrieved_file $RETRIEVAL_FILE
```

Note: Our main retrieval flow assumes you want to use the top-scored passage from your retrieval file (`--ranking_strategy first`).

### Reranking 

To run model with reranking, use the `$RERANKING_FILE` output from `rerank_retrieval_data.py` script.

Then run:
```bash
python eval_lm.py \
  --model_name $MODEL_NAME \
  --dataset_path wikitext \
  --dataset_name wikitext-103-v1 \
  --dataset_split [validation, test] \
  --output_dir $OUTPUT_DIR \
  --stride 4 \
  --max_length 1024 \
  [--model_parallelism] \
  --retrieved_file $RERANK_FILE \
  --ranking_strategy first-rerank \
  --layer -1 \ # For strategies that require a layer
  --num_docs_to_rank 16 \
  --ranking_logprob_past_tokens 16
```

Note: The reranked file doesn't store documents in sorted order, thus `--ranking_strategy first-rerank` will dynamically find the top scoring document.

## Display Results

We can use the script `summarize_results.py` in order to summarize runs and create graphics.

## Acknowledgements 

This code base was forked from [In Context RALM](https://github.com/AI21Labs/in-context-ralm).
