#!/bin/bash
#BSUB -J "atari_conv[1-40]"
#BSUB -q hpc
#BSUB -n 10
#BSUB -W 60:00
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"

#BSUB -o logs/atari/conv_%J_%I.out
#BSUB -e logs/atari/conv_%J_%I.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate atarenv

export MALLOC_TRIM_EVERY=1000

python -u train_array.py \
    --env-type atari \
    --setup conventional \
    --ablations rainbow dqn no_double no_nstep no_per no_dueling no_distrib no_noisy \
    --seeds 10 \
    --seed-start 0 \
    --seed-procs 10