export MODEL_NAME='gpt2'
export OUTPUT_DIR='logs/results-gp2-bert-base'
export RETRIEVAL_FILE='jsons/bert_reranked_wikitext_rql_32_rs_4_topK_16.json'

python3 -m benchmark.eval_lm \
    --model_name $MODEL_NAME \
    --dataset_path wikitext \
    --dataset_name wikitext-103-v1 \
    --dataset_split 'validation' \
    --output_dir $OUTPUT_DIR \
    --stride 4 \
    --max_length 32 \
    --retrieved_file $RETRIEVAL_FILE \
    --ranking_strategy 'colbert' \
    --num_docs_to_rank 16 \
    --ranking_logprob_past_tokens 16 \
    --retrieved_max_length 32 \
    --model_layer 12
