#!/bin/bash
#SBATCH --job-name=tmp                 # Job name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Tasks per node
#SBATCH --cpus-per-task=8              # CPU cores per task
#SBATCH --gres=gpu:l40s:1              # Request 1 L40S GPU
#SBATCH --mem=16G                      # Memory
#SBATCH --output=../logs/%x-%j.out     # Stdout/stderr log (%x=job name, %j=job ID)

# --- Configuration ---

# Activate Python environment
source ../linkbert/bin/activate

# Set the model and task names
export MODEL_NAME="BioLinkBERT-large"
export TASK_NAME="medqa_usmle_hf"

# Set the base model path from Hugging Face
export INITIAL_MODEL_PATH="michiyasunaga/$MODEL_NAME"

# Define the random seeds for each of the 5 runs
readonly SEEDS=(42 123 456 789 1011)

# Define common training parameters
readonly PER_DEVICE_TRAIN_BATCH_SIZE=2
readonly PER_DEVICE_EVAL_BATCH_SIZE=2
readonly GRADIENT_ACCUMULATION_STEPS=32
readonly WEIGHT_DECAY=0.01
readonly ADAM_BETA1=0.9
readonly ADAM_BETA2=0.98
readonly ADAM_EPSILON=1e-6
readonly LEARNING_RATE=3e-5
readonly WARMUP_STEPS=500
readonly NUM_TRAIN_EPOCHS=6
readonly MAX_SEQ_LENGTH=512

# --- Base Directories ---
readonly DATA_DIR="../data/mc/$TASK_NAME"
readonly BASE_RUNS_DIR="../../runs/$TASK_NAME-new"

# --- Function for Training and Evaluation ---

train_and_evaluate() {
    local model_path=$1
    local train_file=$2
    local output_dir=$3
    local save_strategy=$4
    local seed=$5 # Now accepting seed as a parameter

    echo "------------------------------------------------------------------"
    echo "Starting run with the following settings:"
    echo "  Model Path: $model_path"
    echo "  Train File: $train_file"
    echo "  Output Dir: $output_dir"
    echo "  Save Strategy: $save_strategy"
    echo "  Seed: $seed"
    echo "------------------------------------------------------------------"

    mkdir -p "$output_dir/model"

    python3 -u mc/run_multiple_choice.py \
      --model_name_or_path "$model_path" \
      --train_file "$train_file" \
      --validation_file "$DATA_DIR/dev.json" \
      --test_file "$DATA_DIR/test.json" \
      --do_train \
      --do_predict \
      --seed "$seed" \
      --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
      --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
      --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
      --weight_decay $WEIGHT_DECAY \
      --adam_beta1 $ADAM_BETA1 \
      --adam_beta2 $ADAM_BETA2 \
      --adam_epsilon $ADAM_EPSILON \
      --learning_rate $LEARNING_RATE \
      --warmup_steps $WARMUP_STEPS \
      --num_train_epochs $NUM_TRAIN_EPOCHS \
      --max_seq_length $MAX_SEQ_LENGTH \
      --fp16 \
      --save_strategy "$save_strategy" \
      --save_total_limit 1 \
      --evaluation_strategy epoch \
      --output_dir "$output_dir/model" \
      --overwrite_output_dir \
      |& tee "$output_dir/log.txt"
}

# --- Main Execution ---

# Loop over each predefined seed
for seed in "${SEEDS[@]}"; do
    echo "##################################################################"
    echo "### Starting Full Run for Seed: $seed ###"
    echo "##################################################################"

    # Create a unique directory for this seed's results
    SEED_RUNS_DIR="$BASE_RUNS_DIR/seed_$seed"

    # 1. Initial training run that saves the model
    echo "--- Starting Initial Training for Seed: $seed ---"

    # synth_1000, synth_5000,synth_10000,synth_20000
    DATA=synth_1000
    initial_outdir="$SEED_RUNS_DIR/$MODEL_NAME-$DATA-e6"
    train_and_evaluate "$INITIAL_MODEL_PATH" "$DATA_DIR/$DATA.json" "$initial_outdir" "epoch" "$seed"

    # Set the model path for subsequent runs to the output of the first run for this specific seed
    FINE_TUNED_MODEL_PATH="$initial_outdir/model"
    FINE_TUNED_MODEL_PATH=$INITIAL_MODEL_PATH
    # 2. Subsequent training runs on different data subsets
    echo "--- Starting Subsequent Training Runs for Seed: $seed ---"
 
    for train_subset in 1000 2500 5000 7500; do
        subset_outdir="$SEED_RUNS_DIR/$MODEL_NAME-$DATA-train_${train_subset}-e6"
        train_and_evaluate "$FINE_TUNED_MODEL_PATH" "$DATA_DIR/train_${train_subset}.json" "$subset_outdir" "no" "$seed"
    done

    # 3. Final run on the full training data
    echo "--- Starting Final Run on Full Training Data for Seed: $seed ---"
    final_outdir="$SEED_RUNS_DIR/$MODEL_NAME-$DATA-train_full-e6"
    train_and_evaluate "$FINE_TUNED_MODEL_PATH" "$DATA_DIR/train.json" "$final_outdir" "no" "$seed"

    echo "### Completed Full Run for Seed: $seed ###"
done

echo "--- All training runs for all seeds completed ---"