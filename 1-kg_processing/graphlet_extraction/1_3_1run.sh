#!/bin/bash
#SBATCH --job-name=graphlet_counter                # create a short name for your job
#SBATCH --output="graphlet_counter-%j.out"         # %j will be replaced by the slurm jobID
#SBATCH --nodes=1                         # node count
#SBATCH --ntasks=1                        # total number of tasks across all nodes
#SBATCH --cpus-per-task=24                 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G                          # Total amount of RAM requested
#SBATCH --partition=cpu                   # The queue where to submit the job

module load python
module load anaconda3


export PYTHONUNBUFFERED=TRUE


conda activate gt

/data/home/richard.jonker/.conda/envs/gt/bin/python3 graphlet_extraction.py

