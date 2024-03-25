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
#SBATCH -t 24:00:00                       # Run time (hh:mm:ss)
#SBATCH --mail-user=haydenprairie@utexas.edu
#SBATCH --mail-type=all                   # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A MLL                            # Allocation name


# Source conda environment
source $WORK/miniconda3/bin/activate ralm 

# Launch Job

cd $WORK/projects/zero-shot-colbert-retrieval
python3 -m cbralm.zeroshot_rerank \
    --data-dir '' \
    --project-name 'logs/wikitext_rql_32_rs_4_topK_100' \
    --run-name 'bert_reranked_wikitext_rql_32_rs_4_topK_16' \
    --retrieval-corpus 'wikipedia-dpr-100w' \
    --rerank-model 'bert-base-uncased' \
    --topK 16 \
    --batch-size 8 \
