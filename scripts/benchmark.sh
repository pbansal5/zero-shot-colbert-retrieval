python eval_lm.py \
    --model_name gpt2 \
    --dataset_path wikitext \
    --dataset_name wikitext-103-v1 \
    --dataset_split test \
    --output_dir benchmark/bm25 \
    --stride 4 \
    --max_length 1024 \
    --retrieved_file logs/bm25_retrieval
