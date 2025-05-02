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
                    model_format='awq',
                    cache_max_entry_count=0.65,
                    quant_policy=4,
                )


GENERATION_CONFIG = GenerationConfig(
                    max_new_tokens=1000,
                )

BATCH_SIZE = 30_000

PROMPT= """I will provide you with a graphlet, and your task is to generate a biomedical question-answer pair based on the information within the graphlet. \nTo enhance the complexity of the question, aim to incorporate as many hops as possible while maintaining coherence.\n# Instructions:\n1. Analyze the graphlet, identifying how nodes are connected and how they might relate in a biomedical context.\n\t -  Identify three key node types in the graphlet: \n\t\t 1. Question Nodes: These should appear in the question and provide the context for inquiry.\n\t\t 2. Answer Nodes: These should not be explicitly mentioned in the question but must be inferable from the graph structure.\n\t\t 3. Hidden Nodes: These act as logical intermediaries, enabling multi-hop reasoning to reach the answer.\n2. Construct a question that a biomedical expert might ask, ensuring that: \n\t - Ensure the question is phrased naturally as if it were asked in a biomedical research or clinical context. You may mention some graphlet nodes, but do not give away the answer. The question should require multi-step reasoning.\n\t - The answer to the question should be in the graph structure. \n\t- The nodes required to answer the question should not be in the question.\n\t - Ensure scientific relevance, aligning with biomedical terminology and logical reasoning.\n3. Your answer should be a scientifically valid response based on the graphlet. Ensure:\n\t - The response is more than a single word; provide a concise yet informative explanation.\n\t - It should justify the answer by connecting relevant biomedical concepts. \n\t - Use precise biomedical terminology while maintaining clarity.\n4.  After writing the question and answer, you should reflect on the output and improve the QA pair. If there are no improvements to be made, please repeat the Question/Answer.\n\t - Question Evaluation Criteria:\n\t\t - Is the question unambiguous and focused?\n\t\t - Does the question reflect realistic clinical or research scenarios?  \n\t\t - Does the question require integration of multiple concepts?  \n\t\t - Are terms precise, or could they mislead?  \n\t\t - Is the question too easy?\n\t\t - Does the question sound natural, or is it too focused on connections from the graph?\n\t - Answer Evaluation Criteria:\n\t\t - Are all facts correct?\n\t\t - Does the answer address all parts of the question?\n\t\t - Are key connections explained?\n\t\t - Does it avoid unsupported claims?\n\t\t - Are claims supported by pharmacological principles?\n# Example:\n## Analysis of Graphlet:\nGraphlet contains nodes: [Cholera, Contaminated Water, Fecal-Oral Route, Dehydration]\n - Question Node: Cholera\n - Hidden Node: Contaminated Water\n - Answer Node: Fecal-Oral Route\n## Initial QA\n### Question:\n - What is the primary transmission route for infections like cholera?\n### Answer\n - The fecal-oral route is a primary transmission pathway for infections such as cholera. Contaminated food or water sources facilitate the spread of bacteria like Vibrio cholerae, leading to severe dehydration and gastrointestinal distress.\n\n## Reflection\n\n## Final QA\n### Question:\n### Answer:\n\n\nNow, analyze the given graphlet and generate a well-formed biomedical question-answer pair.\nPlease return the final QA pair in json format of {"question", "answer"}.\n'"""
# PROMPT= """I will provide you with a graphlet, and your task is to generate a biomedical question-answer pair based on the information within the graphlet. \nTo enhance the complexity of the question, aim to incorporate as many hops as possible while maintaining coherence.\n# Instructions:\n1. Analyze the graphlet, identifying how nodes are connected and how they might relate in a biomedical context.\n\t -  Identify three key node types in the graphlet: \n\t\t 1. Question Nodes: These should appear in the question and provide the context for inquiry.\n\t\t 2. Answer Nodes: These should not be explicitly mentioned in the question but must be inferable from the graph structure.\n\t\t 3. Hidden Nodes: These act as logical intermediaries, enabling multi-hop reasoning to reach the answer.\nNow, analyze the given graphlet and generate a well-formed biomedical question-answer pair.\nPlease return the final QA pair in JSON format of {\"question\", \"answer\"}.\n"""
if __name__ == '__main__':
    
    with open("last_output.txt", "r") as file:
        START_INDEX = int(file.read().strip())  # Read, strip whitespace, and convert to integer
 

    with open(PATH_TO_CONSTANTS+CONSTANTS['templates']) as f:
        data = [json.loads(line) for line in f]
    END_INDEX = min(len(data), START_INDEX+BATCH_SIZE)
    data = data[START_INDEX:END_INDEX]
    
    
    answer_template = {}
    
    counter  = 0
    
    prompts = []
    for graphlet in data:
        
        # answer_template[counter] = {'id': graphlet['id'],  'graphlet_id': graphlet['graphlet_id'], 'text':""}
        # counter+=1
        prompts.append(f"{PROMPT}\n# Graphlet\n## Nodes:\n{''.join(graphlet['graphlet_text'])}\n## Edges:\n{graphlet['edges']}")

    
    # for iso_code, values in templates_dict.items():
    #     edges = ""
    #     for edge in values['edges']:
    #         edges += f"- ({edge[0]}, {edge[1]})\n"
    #     #     values['edges']
    #     for node in values['graphlets']:
    #         answer_template[counter] = {'template_id': iso_code, 'text':"", 'nodes': node, "edges": edges}
    #         counter+=1
    #         #verify this is okay

    #         prompts.append(f"{PROMPT}\n# Graphlet\n## Nodes:\n{node}\n## Edges:\n{edges}")
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
        data[j]['text'] = i.text
    # answer_dict[j] = {'answer': i.text, 'nodes': prompts[i]}
    print(f"Total tokens: {total_tokens}, tps:  {total_tokens/response_time}", flush=True)
    print(f"Total questions: {len(prompts)}, average answer length:  {total_tokens/len(prompts)}, average time per question: {response_time/len(prompts)}, qpd: {86400* (len(prompts)/response_time)}", flush=True )


    with open(f"{PATH_TO_CONSTANTS}{CONSTANTS['llm_outs']}/DATASET/{START_INDEX:06d}_{END_INDEX:06d}.jsonl", "w") as file:
        for item in data:
            file.write(json.dumps(item) + "\n")
        
    with open(f'last_output.txt', 'w') as fp:
        fp.write(f"{END_INDEX}")
    print(MODEL)
        