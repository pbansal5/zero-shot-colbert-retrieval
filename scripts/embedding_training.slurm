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
#SBATCH -t 36:00:00                       # Run time (hh:mm:ss)
#SBATCH --mail-user=haydenprairie@utexas.edu
#SBATCH --mail-type=all                   # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A MLL                            # Allocation name


# Source conda environment
source $WORK/miniconda3/bin/activate inralm 

# Launch Job
cd $WORK/projects/in-context-ralm
python train_prefix.py \
    --model_name gpt2 \
    --model_type causal \
    --output_file runs/embeddings \
    --retrieved_file logs/bm25_retrieval_validation \
    --score_file benchmark/bm25_validation_oracle/ppls.pkl \
    --training_config training-configs/finegrain_config.json \
    --max_length 256 \
    --embedding \
    --num_tokens 16 \
    --split_tower

