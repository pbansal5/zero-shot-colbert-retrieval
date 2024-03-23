export MODEL_NAME='gpt2'
export PROJECT_NAME='logs/wikitext_rql_32_rs_4_topK_100'
export RUN_NAME='results-baseline-bm25-random'
export RETRIEVAL_FILE='logs/wikitext_rql_32_rs_4_topK_100/run.json'

python3 -m benchmark.eval_lm \
    --model-name $MODEL_NAME \
    --dataset-path wikitext \
    --dataset-name wikitext-103-v1 \
    --dataset-split 'validation' \
    --run-name $RUN_NAME \
    --project-name $PROJECT_NAME \
    --retrieved-file $RETRIEVAL_FILE \
    --ranking-strategy 'random' \
    --stride 4 \
    --max-length 1024 \
    --num-docs-to-rank 16 \
    --ranking-logprob-past-tokens 16 \
    --retrieved-max-length 32 \



#    --ranking-strategy 'colbert' \
#    --model-layer 12
