#!/bin/bash
#SBATCH --job-name=twolayercnn
#SBATCH --output=sbatch_logs/run-%j-twolayercnn.out
#SBATCH --error=sbatch_logs/run-%j-twolayercnn.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition=short

set -x
sacct -j ${SLURM_JOB_ID} --format=User%20,JobID,Jobname%40,partition,state,time,start,nodelist
bash scripts/run_twolayercnn.sh