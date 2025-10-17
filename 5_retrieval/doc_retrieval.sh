#!/bin/bash
#SBATCH --job-name=llama-text-gen                # create a short name for your job
#SBATCH --output="gen%j.out"         # %j will be replaced by the slurm jobID
#SBATCH --nodes=1                         # node count
#SBATCH --ntasks=1                        # total number of tasks across all nodes
#SBATCH --cpus-per-task=16                 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:nvidia-rtx-a6000:1                      # number of gpus per node
#SBATCH --mem=64G                          # Total amount of RAM requested
#SBATCH --partition=gpu                   # The queue where to submit the job

export PYTHONUNBUFFERED=TRUE



module load python
module load cuda


source ../venv-vllm/bin/activate

python 1_relevant_documents.py
