#!/bin/bash
# Create logs/ before submitting: mkdir -p logs

# 5 environments × 8 ablations = 40 array jobs
# Maximum 10 array jobs running simultaneously
# Each array job runs 10 seeds in parallel

#BSUB -J "minatar_setup1[1-40]%10"

#BSUB -q hpc
#BSUB -n 10
#BSUB -W 40:00
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"

#BSUB -o logs/minatar_setup1_%J_%I.out
#BSUB -e logs/minatar_setup1_%J_%I.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate atarenv

export MALLOC_TRIM_EVERY=1000

python -u train_array.py \
    --env-type minatar \
    --setup setup1 \
    --ablations rainbow dqn no_double no_nstep no_per no_dueling no_distrib no_noisy \
    --seeds 10 \
    --seed-procs 10