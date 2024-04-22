#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
# for TACC Lonestar6  nodes
#----------------------------------------------------

#SBATCH -J CBRALM                         # Job name
#SBATCH -o slurmlogs/BM-RALM.o%j          # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e slurmlogs/BM-RALM.e%j          # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gpu-a100-small                 # Queue (partition) name
#SBATCH -N 1                              # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 36:00:00                       # Run time (hh:mm:ss)
#SBATCH --mail-user=haydenprairie@utexas.edu
#SBATCH --mail-type=all                   # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A MLL                            # Allocation name


# Source conda environment
source $WORK/miniconda3/bin/activate inralm 

# Launch Job
cd $WORK/projects/in-context-ralm

export MODEL_NAME='gpt2'
export RERANKED_FILE='bm25_retrieval_llama_zsllms_max_rerank'
export RUN_NAME='bm25_retrieval_llama_zsllms_max_rerank_gpt2'
export NUM_LAYERS=35

for i in $(seq 0 $NUM_LAYERS)
do 
    python eval_lm.py \
        --model_name $MODEL_NAME \
        --dataset_path wikitext \
        --dataset_name wikitext-103-v1 \
        --dataset_split test \
        --output_dir "benchmark/${RUN_NAME}_layer_$i" \
        --stride 4 \
        --max_length 1024 \
        --retrieved_file "logs/${RERANKED_FILE}" \
        --ranking_strategy 'first-rerank' \
        --layer $i
done
