import argparse
import json
import os
import re
from tqdm import tqdm
from datasets import Dataset
from unsloth import FastModel, FastLanguageModel
from unsloth.chat_templates import get_chat_template
import evaluate

# ------------------------------------------------
# Parse arguments
# ------------------------------------------------
parser = argparse.ArgumentParser(description="Run evaluation on all checkpoints in a model directory.")
parser.add_argument("--model_dir", type=str, required=True, help="Path to directory containing checkpoints.")
parser.add_argument("--output_template", type=str, required=True, help="Base name for output files (e.g., tmp → tmp_e1.jsonl).")
args = parser.parse_args()

# ------------------------------------------------
# Load constants and dataset
# ------------------------------------------------
with open("../constants.json") as f:
    CONSTANTS = json.load(f)

PATH_TO_CONSTANTS = "../"
print("Loading data...")
with open(f"{PATH_TO_CONSTANTS}_bioasq/training13b_inflated_clean_wContents_IA.jsonl", "r") as file:
    raw_data = [json.loads(line) for line in file]

GROUND_TRUTHS = [i['ideal_answer'][0] for i in raw_data if i['edition'] == '12B']

def test_gen():
    for i in raw_data:
        if i['edition'] == '12B':
            yield {'conversations': [{'content': i['body'], 'role': 'user'}]}

# ------------------------------------------------
# Evaluation function
# ------------------------------------------------
def run_eval(model_name, output_path):
    print(f"\n\nEvaluating model: {model_name}")
    model, tokenizer = FastModel.from_pretrained(model_name=model_name, max_seq_length=2048, full_finetuning=False)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    FastLanguageModel.for_inference(model)

    dataset = Dataset.from_generator(test_gen)
    dataset = dataset.map(lambda x: {
        "text": tokenizer.apply_chat_template(x["conversations"], add_generation_prompt=True)
    })

    predictions, references = [], []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as outfile:
        for idx, example in tqdm(enumerate(dataset), total=len(dataset)):
            inputs = tokenizer(example["text"], return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=400, do_sample=False)

            prompt_len = inputs["input_ids"].shape[1]
            assistant_output = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

            ref_clean = GROUND_TRUTHS[idx].strip()

            predictions.append(assistant_output)
            references.append(ref_clean)

            outfile.write(json.dumps({
                "id": idx,
                "prediction": assistant_output,
                "reference": ref_clean,
            }) + "\n")

        # Compute metrics
        rouge = evaluate.load("rouge")
        bertscore = evaluate.load("bertscore")

        results_r = rouge.compute(predictions=predictions, references=references)
        results_b = bertscore.compute(predictions=predictions, references=references, lang="en")

        metrics = {
            'model_name': model_name,
            'rouge2': results_r['rouge2'],
            'f1': sum(results_b['f1']) / len(results_b['f1']),
            'precision': sum(results_b['precision']) / len(results_b['precision']),
            'recall': sum(results_b['recall']) / len(results_b['recall']),
        }

        outfile.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics, indent=2))

# ------------------------------------------------
# Main: Evaluate all checkpoints in given dir
# ------------------------------------------------
all_checkpoints = sorted([
    os.path.join(args.model_dir, d)
    for d in os.listdir(args.model_dir)
    if os.path.isdir(os.path.join(args.model_dir, d)) and "checkpoint" in d
], key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1)))

print(f"Found {len(all_checkpoints)} checkpoints in {args.model_dir}")

for idx, checkpoint in enumerate(all_checkpoints, 1):
    output_file = f"{PATH_TO_CONSTANTS}/_llm_outs/eval_outputs_new_qa/{args.output_template}_E{idx}.jsonl"
    run_eval(checkpoint, output_file)
