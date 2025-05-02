from lmdeploy import pipeline, TurbomindEngineConfig,GenerationConfig, PytorchEngineConfig
import json
import os
import time
import sys

PATH_TO_CONSTANTS = "../"
with open(PATH_TO_CONSTANTS+'constants.json') as f:
    CONSTANTS = json.load(f)

MODEL  = CONSTANTS['model_paths']+'Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4-TurboMind'
BACKEND_CONFIG = TurbomindEngineConfig(
                    cache_max_entry_count=0.7,
                    quant_policy=4, #this is new
                    model_format='awq'
                )

GENERATION_CONFIG = GenerationConfig(
                    max_new_tokens=1000,
                )

START_INDEX = 0
END_INDEX=1000


if __name__ == '__main__':
    
    
    with open(PATH_TO_CONSTANTS+CONSTANTS['templates']) as f:
        data = [json.loads(line) for line in f]
    data = data[:END_INDEX]
    
    with open("prompt_templates.json") as f:
        prompts_dict = json.load(f)
    prompts = []
    
    answer_template = {}
    
    counter  = 0
    
    for prompt_id, prompt_template in prompts_dict.items():
        
        for graphlet in data:
            
            answer_template[counter] = {'prompt_id':prompt_id, 'id': graphlet['id'],  'graphlet_id': graphlet['graphlet_id'], 'text':""}
            counter+=1
            prompts.append(f"{prompt_template}\n# Graphlet\n## Nodes:\n{''.join(graphlet['graphlet_text'])}\n## Edges:\n{graphlet['edges']}")
#     break
    t0 = time.time()
    print(f"loading model", flush=True)
    pipe = pipeline(MODEL, backend_config = BACKEND_CONFIG, )
    print(f"model loaded in {time.time() - t0}, running infernce",  flush=True)
    print(len(prompts))
    
    t0 = time.time()
    
    
    response = pipe(prompts, gen_config = GENERATION_CONFIG)
    response_time =  time.time() - t0
    print(f"Response generated {response_time}",  flush=True)

    total_tokens = 0
    for j,i in enumerate(response):

        total_tokens+=i.generate_token_len
        answer_template[j+START_INDEX]['text'] = i.text
    print(f"Total tokens: {total_tokens}, tps:  {total_tokens/response_time}", flush=True)
    print(f"Total questions: {len(prompts)}, average answer length:  {total_tokens/len(prompts)}, average time per question: {response_time/len(prompts)}, qpd: {86400* (len(prompts)/response_time)}", flush=True )

    answer_dict = {k: answer_template[k] for k in list(answer_template.keys())}


    with open(f"{PATH_TO_CONSTANTS}{CONSTANTS['llm_outs']}/prompt_testing/{START_INDEX}_{END_INDEX}.json", 'w') as fp:
        json.dump(answer_dict, fp)
    print(MODEL)
        