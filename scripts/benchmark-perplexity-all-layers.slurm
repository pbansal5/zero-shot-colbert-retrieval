#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
# for TACC Lonestar6  nodes
#----------------------------------------------------

#SBATCH -J ZS-BENCH                       # Job name
#SBATCH -o slurmlogs/ZS-BENCH.o%j         # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e slurmlogs/ZS-BENCH.e%j         # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gpu-a100-small                 # Queue (partition) name
#SBATCH -N 1                              # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 12:00:00                       # Run time (hh:mm:ss)
#SBATCH --mail-user=haydenprairie@utexas.edu
#SBATCH --mail-type=all                   # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A MLL                            # Allocation name


# Source environment
source $WORK/miniconda3/bin/activate ralm 
cd $WORK/projects/zero-shot-colbert-retrieval

# Variables
export MODEL_NAME='gpt2'
export PROJECT_NAME='logs/wikitext_rql_32_rs_4_topK_100'
export RUN_NAME='results-bert-reranked'
export RETRIEVAL_FILE='logs/wikitext_rql_32_rs_4_topK_100/bert_reranked_wikitext_rql_32_rs_4_topK_16/run.json'
export NUM_LAYERS=12

# Launch Job
for i in $(seq 0 $NUM_LAYERS)
do 
    python3 -m benchmark.eval_lm \
        --model-name $MODEL_NAME \
        --dataset-path wikitext \
        --dataset-name wikitext-103-v1 \
        --retrieval-corpus 'wikipedia-dpr-100w' \
        --dataset-split 'test' \
        --run-name "$RUN_NAME-layer-$i" \
        --project-name $PROJECT_NAME \
        --retrieved-file $RETRIEVAL_FILE \
        --ranking-strategy 'colbert' \
        --model-layer $i \
        --stride 4 \
        --max-length 1024 \
        --num-docs-to-rank 16 \
        --ranking-logprob-past-tokens 16 \
        --retrieved-max-length 32
done
