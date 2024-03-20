export MODEL_NAME='gpt2'
export OUTPUT_DIR='logs/baseline'

python3 -m benchmark.eval_lm \
    --model_name $MODEL_NAME \
    --dataset_path wikitext \
    --dataset_name wikitext-103-v1 \
    --dataset_split 'validation' \
    --output_dir $OUTPUT_DIR \
    --stride 4 \
    --max_length 32 \
