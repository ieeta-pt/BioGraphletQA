#!/bin/bash
#SBATCH --job-name=LLM-FT                # create a short name for your job
#SBATCH --output="ft%j.out"         # %j will be replaced by the slurm jobID
#SBATCH --nodes=1                         # node count
#SBATCH --ntasks=1                        # total number of tasks across all nodes
#SBATCH --cpus-per-task=16                 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:nvidia-rtx-a6000:1                      # number of gpus per node
#SBATCH --mem=64G                        # Total amount of RAM requested
#SBATCH --partition=gpu                   # The queue where to submit the job

export PYTHONUNBUFFERED=TRUE



module load python

source ../venv/bin/activate # If you have your venv activated when you submit the job, then you do not need to activate/deactivate

# echo "I m alive"




# python unsloth_trainer_bioasq_new_qa.py \
#     --model_checkpoint "../_model/gemma-3-12b-ft-custom-E3/checkpoint-5619" \
#     --run_name "gemma-3-12b-ft-custom-E3-bioasq-E3-val_new"
# python new_new_eval.py \
#   --model_dir "../_model/_new_models/gemma-3-12b-ft-custom-E3-bioasq-E3-val_new" \
#   --output_template "E3"


# python unsloth_trainer_bioasq_new_qa.py \
#     --model_checkpoint "../_model/gemma-3-12b-ft-custom-E3/checkpoint-5619" \
#     --run_name "gemma-3-12b-ft-custom-E0-bioasq-E3-val_new"  
# # up is the new baseline model
# python new_new_eval.py \
#   --model_dir "../_model/_new_models/gemma-3-12b-ft-custom-E0-bioasq-E3-val_new" \
#   --output_template "E0"


# python unsloth_trainer_bioasq_new_qa.py \
#     --model_checkpoint "../_model/gemma-3-12b-ft-custom-E3/checkpoint-3746" \
#     --run_name "gemma-3-12b-ft-custom-E2-bioasq-E3-val_new"
# python new_new_eval.py \
#   --model_dir "../_model/_new_models/gemma-3-12b-ft-custom-E2-bioasq-E3-val_new" \
#   --output_template "E2"


# python unsloth_trainer_bioasq_new_qa.py \
#     --model_checkpoint "../_model/gemma-3-12b-ft-custom-E3/checkpoint-1873" \
#     --run_name "gemma-3-12b-ft-custom-E1-bioasq-E3-val_new"
# python new_new_eval.py \
#   --model_dir "../_model/_new_models/gemma-3-12b-ft-custom-E1-bioasq-E3-val_new" \
#   --output_template "E1"




python unsloth_trainer_merged.py \
    --model_checkpoint "unsloth/gemma-3-12b-it" \
    --run_name "gemma-3-12b-ft-mixed-E3"
python new_new_eval.py \
  --model_dir "../_model/_new_models/gemma-3-12b-ft-mixed-E3" \
  --output_template "mixed"

deactivate