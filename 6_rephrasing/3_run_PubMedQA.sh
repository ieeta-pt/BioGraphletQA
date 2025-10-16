#!/bin/bash
#SBATCH --job-name=run_pubmedqa      # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Tasks per node
#SBATCH --cpus-per-task=8            # Adjust CPU cores per GPU (tune as needed)
#SBATCH --gres=gpu:l40s:1             # Request 4 A100 GPUs (use up to 8)
#SBATCH --mem=16G                    # Memory (adjust; node has ~980GB)
#SBATCH --output=logs/pubmedqa2.out       # Stdout/stderr log (%x=job name, %j=job ID)

# --- Load required modules (adjust for your cluster environment) ---
# module purge
#--partition=batch             # Use the batch partition

module load cuda
# module load gcc/11.2   # if your Python build depends on it

# --- Activate virtual environment (created with uv or venv) ---
source ../.venv/bin/activate

# uv pip install rank_bm25
# --- Debug info ---
echo "Running on node: $(hostname)"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"


python3 qa_rephrasing_PubMedQA.py
