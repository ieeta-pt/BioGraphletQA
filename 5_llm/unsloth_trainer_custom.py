
from unsloth import FastModel
import torch
import json
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
from unsloth.chat_templates import train_on_responses_only

import os

os.environ["WANDB_ENTITY"] = "bitua"
os.environ["WANDB_PROJECT"] = "Synthetic-KGQA-model-FT"
os.environ["WANDB_LOG_MODEL"] = "gemma-3-12b-ft-custom-E3"





#define some constants (file paths)
with open("../constants.json") as f:
    CONSTANTS = json.load(f)
    print(CONSTANTS)

PATH_TO_CONSTANTS = "../"


print("Preparing model")

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-12b-it",
    max_seq_length = 1024, # Choose any for long context!
    # load_in_4bit = True,  # 4 bit quantization to reduce memory
    # load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 42,
)



tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)



print("preparing data")

with open(f"{PATH_TO_CONSTANTS}{CONSTANTS['llm_outs']}/FILTER_DATASET/filtered_ds.jsonl", "r") as file:
    raw_data = [json.loads(line) for line in file]  
def gen():
    for i in raw_data:
        yield {'conversations': [
            {'content':i['question'], 'role':'user'},
            {'content': i['answer'], 'role':'assistant'}
             ]}
def apply_chat_template(examples):
    texts = tokenizer.apply_chat_template(examples["conversations"])
    return { "text" : texts }        
dataset = Dataset.from_generator(gen)

dataset = dataset.map(apply_chat_template, batched = True)

print("preparing trainer")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 32,
        # gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 3, # Set this for 1 full training run.
        # max_steps = 100,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "wandb",
        logging_steps = 1, # Change if needed
        run_name = "gemma-3-12b-ft-custom-E3", # (Optional)
        output_dir = f"{PATH_TO_CONSTANTS}_model/gemma-3-12b-ft-custom-E3",
        save_strategy = "epoch",
        # save_steps = 50,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

trainer_stats = trainer.train(resume_from_checkpoint = False)


