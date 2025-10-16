# answer_node_in_question
I will give you a question and answer pair, your job is to tell me if the answer is present within the question. The answer should have a specific entity not mentioned in the question, if this does not happen return true. Similarly if the answer is a yes/no question or a description about the entities present in the question return true. Please respond only with JSON:
{"justification": justification, "answer_node_in_question": true/false}.
The justification should be a single string, and "answer_node_in_question" must only be a boolean.

# question_mentions_graphlet_terms
I will give you a question, your job is to tell me if the question mentions any terms that could be related to a graphlet. Please respond only with JSON: 
 {"justification": justification, "question_mentions_graphlet_terms: true/false}.
The justification should be a single string, and "question_mentions_graphlet_terms" must only be a boolean.

# answer_mentions_graphlet_terms
I will give you an answer to a question, your job is to tell me if the answer mentions any terms that could be related to a graphlet. Please respond only with JSON: 
 {"justification": justification, "answer_mentions_graphlet_terms: true/false}.
The justification should be a single string, and "answer_mentions_graphlet_terms" must only be a boolean.

# scientifically_accurate_question
I will give you a question, your job is to tell me if the question is scientifically accurate and makes sense from a biological standpoint. The question should sound like an expert is asking it. Further the question should not be trivial. Please respond only with JSON:
{"justification": justification, "scientifically_accurate_question": true/false}.
The justification should be a single string, and "scientifically_accurate_question" must only be a boolean.

# scientifically_accurate_answer
I will give you an answer to a question, your job is to tell me if the answer is scientifically accurate and makes sense from a biological standpoint. The answer should not be one worded, and be a relatively complete answer, explaining justifications. Please respond only with JSON:
{"justification": justification, "scientifically_accurate_answer": true/false}.
The justification should be a single string, and "scientifically_accurate_answer" must only be a boolean.

# answers_question
I will give you a question/answer pair, your job is to tell me if the answer correctly answers the question, and the answer is complete, not lacking in any additional knowledge. Please respond only with JSON: 
 {"justification": justification, "answers_question": true/false}.
The justification should be a single string, and "answers_question" must only be a boolean.
