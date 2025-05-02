#!/bin/bash
#SBATCH --job-name=LLM-FT                # create a short name for your job
#SBATCH --output="ft%j.out"         # %j will be replaced by the slurm jobID
#SBATCH --nodes=1                         # node count
#SBATCH --ntasks=1                        # total number of tasks across all nodes
#SBATCH --cpus-per-task=16                 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:nvidia-rtx-a6000:1                      # number of gpus per node
#SBATCH --mem=128G                          # Total amount of RAM requested
#SBATCH --partition=gpu                   # The queue where to submit the job

export PYTHONUNBUFFERED=TRUE



module load python

source ../venv/bin/activate # If you have your venv activated when you submit the job, then you do not need to activate/deactivate


python unsloth_trainer.py

deactivate