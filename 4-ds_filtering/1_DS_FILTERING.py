from lmdeploy import pipeline, TurbomindEngineConfig,GenerationConfig, PytorchEngineConfig
import json
import os
import time
import sys




PATH_TO_CONSTANTS = "../"
with open(PATH_TO_CONSTANTS+'constants.json') as f:
    CONSTANTS = json.load(f)
    
    
MODEL  =  CONSTANTS['model_paths']+'Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4-TurboMind'
BACKEND_CONFIG = TurbomindEngineConfig(
                    model_format='awq',
                    cache_max_entry_count=0.65,
                    quant_policy=4,

                )


GENERATION_CONFIG = GenerationConfig(
                    max_new_tokens=1000,
                )

BATCH_SIZE = 50_000



PROMPT= """Evaluate the following question answer pair, first analyze the question, identifying different entities. Then evaluate the various connections between these nodes and identify if the question makes sense from a biomedical standpoint. \n\n After this take the question and try to answer it correctly, being the most scientifically correct.\n\n Finally compare your answer to the answer I provide and tell me if it is scientifically accurate, and completely answers the question. \n\n Present your findings in a json string: "{question_reasoning: "", valid_question: true/false, my_answer:"", answer_reasoning:"", original_answer_valid:true/false }"\n\n The fields original_answer_valid and valid_questions must be boolean, the field must be valid json, no comments.\n\n """

if __name__ == '__main__':
    
    
    
    with open("last_output.txt", "r") as file:
        START_INDEX = int(file.read().strip())  # Read, strip whitespace, and convert to integer


    
    with open(f"{PATH_TO_CONSTANTS}{CONSTANTS['llm_outs']}/DATASET/cleaned_dataset.jsonl") as f:
        ds = [json.loads(line) for line in f]  
        
    END_INDEX = min(len(ds), START_INDEX+BATCH_SIZE)

    print(f"Now processing for batch {START_INDEX}:{END_INDEX}, out of {len(ds)}")
    
    ds = ds[START_INDEX:END_INDEX]
    
            
    prompts = []
    for item in ds:
        prompts.append(f"{PROMPT}Question:{item['question']}\nAnswer:\n{item['answer']}")
    print(f"legnth of the prompts: {len(prompts)}, dataset length: {len(ds)}")

    t0 = time.time()
    print(f"loading model", flush=True)
    pipe = pipeline(MODEL, backend_config = BACKEND_CONFIG, )
    print(f"model loaded in {time.time() - t0}, running infernce",  flush=True)
    
    t0 = time.time()
    
    response = pipe(prompts, gen_config = GENERATION_CONFIG)
    response_time =  time.time() - t0
    print(f"Response generated {response_time}",  flush=True)

    total_tokens = 0
    for j,i in enumerate(response):

        total_tokens+=i.generate_token_len
        ds[j]['filtering_text'] = i.text
        
    print(f"Total tokens: {total_tokens}, tps:  {total_tokens/response_time}", flush=True)
    print(f"Total questions: {len(prompts)}, average answer length:  {total_tokens/len(prompts)}, average time per question: {response_time/len(prompts)}, qpd: {86400* (len(prompts)/response_time)}", flush=True )


    with open(f"{PATH_TO_CONSTANTS}{CONSTANTS['llm_outs']}/FILTER_DATASET/{START_INDEX:06d}_{END_INDEX:06d}.jsonl", "w") as file:
        for item in ds:
            file.write(json.dumps(item) + "\n")
        
    with open(f'last_output.txt', 'w') as fp:
        fp.write(f"{END_INDEX}")


    print(MODEL)
