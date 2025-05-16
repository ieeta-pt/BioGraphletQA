from lmdeploy import pipeline, TurbomindEngineConfig,GenerationConfig, PytorchEngineConfig
import json
import os
import time
import sys


PATH_TO_CONSTANTS = "../"
with open(PATH_TO_CONSTANTS+'constants.json') as f:
    CONSTANTS = json.load(f)


MODEL = CONSTANTS['model_paths']+'Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4-TurboMind'
BACKEND_CONFIG = TurbomindEngineConfig(
                    model_format='awq',
                    cache_max_entry_count=0.7,
                    quant_policy=4
                )


GENERATION_CONFIG = GenerationConfig(
                    max_new_tokens=1000,
                )

if __name__ == '__main__':
    
    
    with open(f"{PATH_TO_CONSTANTS}{CONSTANTS['llm_outs']}/prompt_eval/prompts_1000.json") as f:
        prompts_dict = json.load(f)
    
    answer_template = {}
    
    counter  = 0
    keys = ['answer_node_in_question', 'question_mentions_graphlet_terms', 'answer_mentions_graphlet_terms', 'scientifically_accurate_question', 'scientifically_accurate_answer', 'answers_question']
    prompts = []
    for key, value in prompts_dict.items():
        for k in keys: 
            prompts.append(value[k])
            answer_template[counter] = {"text":"", "type":k, "id": value['id'],
            "prompt_id": value['prompt_id'],
            "graphlet_id": value['graphlet_id'],}
            counter+=1
    
    
    t0 = time.time()
    print(f"loading model", flush=True)
    pipe = pipeline(MODEL, backend_config = BACKEND_CONFIG, )
    print(f"model loaded in {time.time() - t0}, running infernce",  flush=True)
    print(len(prompts))
        
    t1 = time.time()
    
    response = pipe(prompts, gen_config = GENERATION_CONFIG)
    response_time =  time.time() - t1
    print(f"Response generated {response_time}",  flush=True)

    total_tokens = 0
    for j,i in enumerate(response):
        total_tokens+=i.generate_token_len
        answer_template[j]['text'] = i.text
    print(f"Total tokens: {total_tokens}, tps:  {total_tokens/response_time}", flush=True)
    print(f"Total Prompts: {len(prompts)}, average out length:  {total_tokens/len(prompts)}, average time per prompt: {response_time/len(prompts)}, ppd: {86400* (len(prompts)/response_time)}", flush=True )

    answer_dict = {k: answer_template[k] for k in list(answer_template.keys())}

    with open(f"{PATH_TO_CONSTANTS}{CONSTANTS['llm_outs']}/prompt_eval/prompt_evaluation_qa_nemotron_new_prompts.json", 'w') as fp:
        json.dump(answer_dict, fp)
    print(MODEL)
        