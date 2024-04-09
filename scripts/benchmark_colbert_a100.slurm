#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
# for TACC Lonestar6  nodes
#----------------------------------------------------

#SBATCH -J CBRALM                         # Job name
#SBATCH -o slurmlogs/BM-RALM.o%j          # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e slurmlogs/BM-RALM.e%j          # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gpu-a100                       # Queue (partition) name
#SBATCH -N 1                              # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00                       # Run time (hh:mm:ss)
#SBATCH --mail-user=haydenprairie@utexas.edu
#SBATCH --mail-type=all                   # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A MLL                            # Allocation name


# Source conda environment
source $WORK/miniconda3/bin/activate inralm 

# Launch Job
cd $WORK/projects/in-context-ralm

export MODEL_NAME='gpt2-medium'
export RERANKED_FILE='bm25_retrieval_gpt2_medium_colbert_rerank'
export RUN_NAME='bm25_retrieval_gpt2_medium_colbert_rerank_gpt2_medium'
export NUM_LAYERS=23

for i in $(seq 0 $(($NUM_LAYERS / 3)))
do 
    (export CUDA_VISIBLE_DEVICES=0; python eval_lm.py \
                                        --model_name $MODEL_NAME \
                                        --dataset_path wikitext \
                                        --dataset_name wikitext-103-v1 \
                                        --dataset_split test \
                                        --output_dir "benchmark/${RUN_NAME}_layer_$i" \
                                        --stride 4 \
                                        --max_length 1024 \
                                        --retrieved_file "logs/${RERANKED_FILE}" \
                                        --ranking_strategy 'first-rerank' \
                                        --layer $((3*$i)) &)

    if [$((3*$i + 1)) -lt $NUM_LAYERS]; then
        (export CUDA_VISIBLE_DEVICES=1; python eval_lm.py \
                                            --model_name $MODEL_NAME \
                                            --dataset_path wikitext \
                                            --dataset_name wikitext-103-v1 \
                                            --dataset_split test \
                                            --output_dir "benchmark/${RUN_NAME}_layer_$i" \
                                            --stride 4 \
                                            --max_length 1024 \
                                            --retrieved_file "logs/${RERANKED_FILE}" \
                                            --ranking_strategy 'first-rerank' \
                                            --layer $((3*$i + 1)) &)
    fi

    if [$((3*$i + 2)) -lt $NUM_LAYERS]; then
        (export CUDA_VISIBLE_DEVICES=2; python eval_lm.py \
                                            --model_name $MODEL_NAME \
                                            --dataset_path wikitext \
                                            --dataset_name wikitext-103-v1 \
                                            --dataset_split test \
                                            --output_dir "benchmark/${RUN_NAME}_layer_$i" \
                                            --stride 4 \
                                            --max_length 1024 \
                                            --retrieved_file "logs/${RERANKED_FILE}" \
                                            --ranking_strategy 'first-rerank' \
                                            --layer $((3*$i + 2)) &)
    fi
    wait
done
