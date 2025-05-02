import argparse
import json
import os

import torch
from datasets import Dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------
# Argument Parser
# ---------------------------------------------
parser = argparse.ArgumentParser(description="Fine-tune a model with Unsloth + TRL.")
parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to model checkpoint')
parser.add_argument('--run_name', type=str, required=True, help='WandB run name')
args = parser.parse_args()

# ---------------------------------------------
# WandB Setup
# ---------------------------------------------
os.environ["WANDB_ENTITY"] = "bitua"
os.environ["WANDB_PROJECT"] = "Synthetic-KGQA-model-FT-new-qa"
os.environ["WANDB_LOG_MODEL"] = args.run_name

# ---------------------------------------------
# Load Constants
# ---------------------------------------------
with open("../constants.json") as f:
    CONSTANTS = json.load(f)
    print(CONSTANTS)

PATH_TO_CONSTANTS = "../"

# ---------------------------------------------
# Load Model
# ---------------------------------------------
print("Preparing model...")
model, tokenizer = FastModel.from_pretrained(
    model_name=args.model_checkpoint,
    max_seq_length=2048,
    full_finetuning=False,
)


if args.model_checkpoint.startswith("../"):
    model = model.merge_and_unload()

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    random_state=42,
)

tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

# ---------------------------------------------
# Load and Prepare Dataset
# ---------------------------------------------
print("Preparing data...")
with open(f"{PATH_TO_CONSTANTS}_bioasq/training13b_inflated_clean_wContents_IA.jsonl", "r") as file:
    raw_data = [json.loads(line) for line in file]

def train_gen():
    for i in raw_data:
        if i['edition'] != '12B':
            yield {'conversations': [
                {'content': i['body'], 'role': 'user'},
                {'content': i['ideal_answer'][0], 'role': 'assistant'}
            ]}

def apply_chat_template(examples):
    texts = tokenizer.apply_chat_template(examples["conversations"])
    return {"text": texts}

train_dataset = Dataset.from_generator(train_gen)
train_dataset = train_dataset.map(apply_chat_template, batched=True)

# ---------------------------------------------
# Trainer Setup
# ---------------------------------------------
print("Preparing trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=1,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="wandb",
        logging_steps=1,
        run_name=args.run_name,
        output_dir=f"{PATH_TO_CONSTANTS}_model/_new_models/{args.run_name}",
        save_strategy="epoch",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)

# ---------------------------------------------
# Start Training
# ---------------------------------------------
trainer_stats = trainer.train(resume_from_checkpoint=False)
