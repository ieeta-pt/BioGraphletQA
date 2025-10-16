import os
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import nltk

# Make sure you have a tokenizer
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import json

import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import re
import heapq

import random
random.seed(42)

from joblib import Parallel, delayed
# import multiprocessing as mp
# mp.set_start_method("spawn", force=True)

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

def get_top_k_for_query(query_tokens, bm25, k):
    """
    Function to get top-k document indices for a single tokenized query.
    """
    doc_scores = bm25.get_scores(query_tokens)
    top_k_idx = heapq.nlargest(k, range(len(doc_scores)), key=doc_scores.__getitem__)
    return top_k_idx

def main():
    # --- 1. Define the Model ID ---
    model_id = "Qwen/Qwen3-32B-AWQ"


    # --- 2. Load the Tokenizer ---
    print(f"Loading tokenizer for '{model_id}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        exit()
    print("Tokenizer loaded successfully.")


    data_path = "../_retrieval/merged_full.jsonl"
    PATH_TO_CONSTANTS = "../"

    with open(data_path) as f:
        data = [json.loads(x) for x in f.readlines()]
        
        
    with open(f"{PATH_TO_CONSTANTS}_med_qa/data_clean/questions/US/train.jsonl", "r") as file:
        med_mcqa_data = [json.loads(line) for line in file]


    data = data[:20_000]
    random.shuffle(med_mcqa_data)
    med_mcqa_data = med_mcqa_data[:1_000]



    # ----- bm25 lookup
    med_question = [doc['question']+" \na) "+doc['options']['A']+" \nb) "+doc['options']['B']+" \nc) "+doc['options']['C']+" \nd) "+doc['options']['D']+" \ne) "+doc['options']['E']
                    for doc in med_mcqa_data]
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in med_question]

    custom_docs = [doc['query_text'] for doc in data]

    tokenized_queries = [word_tokenize(q.lower()) for q in custom_docs]

    bm25 = BM25Okapi(tokenized_corpus)


    k = 5

    # ----- 4. Execute in Parallel -----
    print(f"Retrieving top {k} documents for {len(custom_docs)} queries in parallel...")
    all_top_k_indices = Parallel(n_jobs=-1)(delayed(get_top_k_for_query)(query_tokens, bm25, k) for query_tokens in tqdm(tokenized_queries))

    # --- 4. Define Prompts and Map to Data ---
    print("Formatting prompts...")
    chat_prompts = []
    prompt_to_data_map = {}

    for i in range(len(custom_docs)):
        # Get the pre-computed top k indices for the current query
        top_k_idx = all_top_k_indices[i]

        # Retrieve the original question (query)
        qa = custom_docs[i]

        # Process snippets associated with the original query
        snippets = [snippet for x in data[i]['processed_documents'] if x['relevant'] for snippet in x['snippets']]

        for j in range(len(snippets)):
            if isinstance(snippets[j], dict):
                snippets[j] = snippets[j]['text']

        # Get the text of the most similar documents found by BM25
        similar_docs = [med_question[idx] for idx in top_k_idx]


        conversation = [
                {
                    "role": "system",
                    "content": "You are a biomedical expert specializing in information retrieval and information extraction."
                },
                {
                    "role": "user",
                    "content": f"""Your role is to analyze the Question Answer pair, as well as the snippets provided.
Rewrite the Question Answer pair, as a Multiple Choice Question, using the factually correct snippets, in the style of the example from MedQA.

Instructions:
1. The new MCQ must be supported by the knowledge in the snippets, but must not reference the snippets directly.
2. The style and length should closely follow the MedMCQA examples.
3. Provide 5 options (a–e), with only one correct answer.
4. The reasoning section must:
   - Be **standalone** (do not say “as shown in the snippets” or “according to the text”).
   - Reason over each answer in order.
   - Explain **why the correct answer is correct**.
   - Explain **why each incorrect option is wrong**, in a way that would make sense on its own to someone without the snippets.
   - Use general biomedical knowledge and logical connections derived from the snippets, but stated as if it’s background knowledge.

QA: {qa}

Snippets:
{'\n\n'.join(snippets)}

Example MedQA pairs:
{'\n\n'.join(similar_docs)}

Output format:
{{
  "question": question,
  "opa": option_a,
  "opb": option_b,
  "opc": option_c,
  "opd": option_d,
  "ope": option_e,
   "reasoning": "standalone explanation covering all options",
  "cop": "the correct option a,b,c,d,e"
}}

Rephrase the orignal QA, using the ground truth snippets as a MCQA like the examples.
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
        prompt_to_data_map[formatted_prompt] = {
                'id': data[i]['id'],
                'qa': qa,
                'medQA_similar': similar_docs,
                'snippets': snippets,
            }


    print(f"Created {len(chat_prompts)} prompts to process.")
    print("-------------------------------------\n")


    # --- 5. Run vLLM Inference with Regeneration Loop ---

    # LLM Initialization
    print("Initializing the vLLM engine...")
    start_time = time.time()
    try:
        llm = LLM(
            model=model_id,
            download_dir="/home/id.aau.dk/kr75cs/hf_models",
            seed=42,
            trust_remote_code=True,
            max_model_len=8192,
            gpu_memory_utilization=0.95,
            enforce_eager = True
        )
    except Exception as e:
        print(f"An error occurred during LLM initialization: {e}")
        exit()
    end_time = time.time()
    print(f"LLM initialization took: {end_time - start_time:.2f} seconds.")
    # Sampling Parameters
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=2048)


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
            if mapping['id'] not in qa_results:
                qa_results[mapping['id']] = mapping
                qa_results[mapping['id']]['status'] = 'pending'
                qa_results[mapping['id']]['medQA_rephrase'] = {}
            is_valid = False
            if finish_reason == 'stop':
                # Basic handling for thought/reasoning blocks before the final JSON
                if '</think>' in generated_text:
                    final_text = generated_text.split('</think>', 1)[-1]
                else:
                    final_text = generated_text

                valid_json, parsed_data = extract_json_from_promt(final_text)
                required_keys = [
                    'question',
                    'opa', 'opb', 'opc', 'opd', 'ope',
                    'reasoning', 'cop'
                ]

                if valid_json and all(key in parsed_data for key in required_keys):
                    # Append the processed document data
                    qa_results[mapping['id']]['medQA_rephrase'] = parsed_data
                    qa_results[mapping['id']]['status'] = 'success' # Mark as successful
                    is_valid = True

            if not is_valid:
                print(f"Prompt for id '{mapping['id']}' failed validation. Queuing for retry.")
                prompts_for_next_round.append(original_prompt)

        prompts_to_process = prompts_for_next_round
        retry_count += 1

    # --- Handle Prompts That Failed All Retries ---
    if prompts_to_process:
        print(f"\n--- Failed to process {len(prompts_to_process)} prompts after {MAX_RETRIES} retries ---")
        for failed_prompt in prompts_to_process:
            mapping = prompt_to_data_map[failed_prompt]
            qa_item = mapping['qa']

            if mapping['id'] in qa_results:
                qa_results[mapping['id']]['status'] = 'failed_after_retries'
    output_filename = "../_retrieval/20_000_qwen3_MedQA_rephrase.jsonl"


    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(qa_results, f)

    print("========================================")
    print(f"Processing finished. Results saved to {output_filename}")

if __name__ == "__main__":
    main()