import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import re

def extract_json_from_promt(text):
    """
    Extracts the last JSON object from a string.
    """
    # This regex finds all occurrences of JSON objects (from '{' to '}')
    matches = re.findall(r"\{[\s\S]*\}", text)
    if matches:
        last_json_str = matches[-1]
        try:
            # Attempt to parse the found string into a JSON object
            parsed_json = json.loads(last_json_str, strict=False)
            return True, parsed_json
        except json.JSONDecodeError:
            return False, "Failed to decode JSON from the model's output."
    else:
        return False, "No JSON object found in the output."

# --- 1. Define the Model ID ---
# model_id = "QuantTrio/Qwen3-30B-A3B-Thinking-2507-AWQ"
model_id = "Qwen/Qwen3-32B-AWQ"
# model_id = "cpatonn/Llama-3_3-Nemotron-Super-49B-v1_5-AWQ"


# --- 2. Load the Tokenizer ---
print(f"Loading tokenizer for '{model_id}'...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()
print("Tokenizer loaded successfully.")

# --- 3. Load Data ---
with open("/data/home/richard.jonker/storage/work/synthetic-kgqa/_retrieval/shuffled_data_qa.jsonl") as f:
    data = [json.loads(x) for x in f.readlines()]

with open("/data/home/richard.jonker/storage/work/synthetic-kgqa/_retrieval/last_index.txt") as f:
    last_index = int(f.readline().strip())

# Using a smaller subset for demonstration purposes if needed
data = data[last_index:last_index+5000]

# --- 4. Define Prompts and Map to Data ---
print("Formatting prompts...")
chat_prompts = []
prompt_to_data_map = {}

for item in data:
    qa_text = str(item['query_text'])
    # Assuming 'bm25' is a list of documents
    for doc in item.get('bm25', []):
        conversation = [
            {
                "role": "system",
                "content": "You are a biomedical expert specializing in information retrieval and information extraction."
            },
            {
                "role": "user",
                "content": f"""Your role is to analyze the Question Answer pair, as well as the single document provided. 
You need to identify if the document is relevant, and if it is, extract the relevant parts of it, directly quoting the full relevant snippets.

QA: {qa_text}
Document:
ID: {doc['id']}, {doc['text']}

Output format: 
{{
  "documents": [
    {{
      "id": "{doc['id']}",
      "relevant": true/false,
      "snippets": []
    }}
  ]
}}
"""
            }
        ]

        # Format the prompt using the tokenizer's chat template
        formatted_prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        chat_prompts.append(formatted_prompt)

        # Map the formatted prompt back to its source question and document
        prompt_to_data_map[formatted_prompt] = {
            'qa': item,
            'doc': doc
        }
print(f"Created {len(chat_prompts)} prompts to process.")
print("-------------------------------------\n")


# --- 5. Run vLLM Inference with Regeneration Loop ---

# Sampling Parameters
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=2048)

# LLM Initialization
print("Initializing the vLLM engine...")
start_time = time.time()
try:
    llm = LLM(
        model=model_id,
        download_dir="/beegfs/client/default/dl-models/hf_models",
        seed=42,
        trust_remote_code=True,
        max_model_len=8192, #8192
        gpu_memory_utilization=0.95,
        enforce_eager = True
    )
except Exception as e:
    print(f"An error occurred during LLM initialization: {e}")
    exit()
end_time = time.time()
print(f"LLM initialization took: {end_time - start_time:.2f} seconds.")

# --- Regeneration Loop ---
MAX_RETRIES = 3
prompts_to_process = list(chat_prompts)
retry_count = 0
qa_results = {}

while prompts_to_process and retry_count < MAX_RETRIES:
    print(f"\n--- Starting Generation Round {retry_count + 1} ({len(prompts_to_process)} prompts) ---")
    outputs = llm.generate(prompts_to_process, sampling_params)
    prompts_for_next_round = []

    for output in outputs:
        original_prompt = output.prompt
        generated_text = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason

        mapping = prompt_to_data_map[original_prompt]
        qa_item = mapping['qa']
        doc = mapping['doc']
        
        # Use a unique ID from the QA item if available, otherwise fallback to query_text
        qa_id = qa_item.get('id', qa_item['query_text'])

        if qa_id not in qa_results:
            qa_results[qa_id] = {
                'query_text': qa_item['query_text'],
                'documents': [],
                'status': 'pending' # Default status
            }

        is_valid = False
        if finish_reason == 'stop':
            # Basic handling for thought/reasoning blocks before the final JSON
            if '</think>' in generated_text:
                final_text = generated_text.split('</think>', 1)[-1]
            else:
                final_text = generated_text

            valid_json, parsed_data = extract_json_from_promt(final_text)
            if valid_json and 'documents' in parsed_data:
                # Append the processed document data
                qa_results[qa_id]['documents'].append(parsed_data['documents'][0])
                qa_results[qa_id]['status'] = 'success' # Mark as successful
                is_valid = True

        if not is_valid:
            print(f"Prompt for doc_id '{doc['id']}' failed validation. Queuing for retry.")
            prompts_for_next_round.append(original_prompt)

    prompts_to_process = prompts_for_next_round
    retry_count += 1
        
# --- Handle Prompts That Failed All Retries ---
if prompts_to_process:
    print(f"\n--- Failed to process {len(prompts_to_process)} prompts after {MAX_RETRIES} retries ---")
    for failed_prompt in prompts_to_process:
        mapping = prompt_to_data_map[failed_prompt]
        qa_item = mapping['qa']
        qa_id = qa_item.get('id', qa_item['query_text'])
        # Mark these as failed in the results dictionary
        if qa_id in qa_results:
            qa_results[qa_id]['status'] = 'failed_after_retries'


# --- [BUG FIX] Merge Results and Save ---
print("\nMerging processed results with original data...")
final_output_data = []
for original_item in data:
    # Find the corresponding results using the same unique ID
    qa_id = original_item.get('id', original_item['query_text'])
    
    # Create a copy to avoid modifying the original list in-place
    updated_item = original_item.copy()

    if qa_id in qa_results:
        # If results were successfully generated, add them
        if qa_results[qa_id]['status'] == 'success':
            updated_item['processed_documents'] = qa_results[qa_id]['documents']
            updated_item['processing_status'] = 'success'
        else:
            # If processing failed after all retries, mark it as such
            updated_item['processing_status'] = 'failed'
            updated_item['processed_documents'] = [] # Add empty list for consistency
    else:
        # If the item was never processed (e.g., had no documents to begin with)
        updated_item['processing_status'] = 'not_processed'
    
    final_output_data.append(updated_item)


# Save the final, merged data to a new file
output_filename = f"/data/home/richard.jonker/storage/work/synthetic-kgqa/_retrieval/outputs/{last_index}_{last_index+5000}_qwen3_qa.jsonl"
with open(output_filename, "w", encoding="utf-8") as f:
    for record in final_output_data:
        json_line = json.dumps(record)
        f.write(json_line + "\n")
        

with open("/data/home/richard.jonker/storage/work/synthetic-kgqa/_retrieval/last_index.txt", "w") as f:
    f.write(str(last_index+5000) + "\n")

print("========================================")
print(f"Processing finished. Results saved to {output_filename}")