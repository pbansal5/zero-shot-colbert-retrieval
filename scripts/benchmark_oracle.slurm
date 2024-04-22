#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
# for TACC Lonestar6  nodes
#----------------------------------------------------

#SBATCH -J CBRALM                         # Job name
#SBATCH -o slurmlogs/ZS-RERANK.o%j        # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e slurmlogs/ZS-RERANK.e%j        # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gpu-a100-small                 # Queue (partition) name
#SBATCH -N 1                              # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 18:00:00                       # Run time (hh:mm:ss)
#SBATCH --mail-user=haydenprairie@utexas.edu
#SBATCH --mail-type=all                   # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A MLL                            # Allocation name


# Source conda environment
source $WORK/miniconda3/bin/activate inralm 

# Launch Job
cd $WORK/projects/in-context-ralm
python eval_lm.py \
    --model_name gpt2 \
    --dataset_path wikitext \
    --dataset_name wikitext-103-v1 \
    --dataset_split validation \
    --stride 4 \
    --max_length 1024 \
    --output_dir benchmark/bm25_validation_oracle \
    --retrieved_file logs/bm25_retrieval_validation \
    --ranking_strategy oracle \
    --num_docs_to_rank 16
