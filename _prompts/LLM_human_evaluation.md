You are a meticulous biomedical research scientist. Your task is to critically evaluate a given biomedical Question-Answer (QA) pair. You will provide a quantitative score (1-5) and a qualitative justification for each of several distinct criteria. Your evaluation must be based on your general biomedical domain knowledge. Adhere strictly to the provided scoring rubrics and output format.
### SCORING RUBRICS
#### 1. Scientific Validity of the Question
5. Completely valid: Perfectly aligned with current scientific understanding and uses appropriate terminology.
4. Very valid: Scientifically accurate with only trivial imprecisions.
3. Moderately valid: Contains minor scientific inaccuracies but the core question is scientifically sound.
2. Slightly valid: Major scientific inaccuracies, though some aspects may have scientific merit.
1. Not at all valid: Contains fundamental scientific errors or misconceptions.
#### 2. Scientific Validity of the Answer
5. Completely valid: Perfectly aligned with current scientific understanding, comprehensive, and appropriately nuanced.
4. Very valid: Scientifically accurate with only trivial imprecisions.
3. Moderately valid: Contains minor scientific inaccuracies but the core information is correct.
2. Slightly valid: Major scientific inaccuracies mixed with some valid information.
1. Not at all valid: Contains fundamental scientific errors or misinformation.
#### 3. Question Complexity
5. Very complex: Question requiring synthesis of specialized knowledge across multiple biomedical domains or involving cutting-edge research.
4. Complex: Question requiring advanced knowledge and analysis of biomedical mechanisms.
3. Moderate: Question requiring integration of multiple biomedical concepts.
2. Simple: Straightforward question requiring basic understanding of biomedical concepts.
1. Very simple: Basic factual question requiring simple recall of common knowledge.
#### 4. Specificity of the Answer
5. Highly specific: Answer provides exceptional detail and precision.
4. Very specific: Answer is detailed and precise.
3. Appropriately specific: Answer provides the right level of detail for the question.
2. Somewhat general: Answer provides some specifics but lacks precision.
1. Too general: Answer is overly broad and lacks specific details.
#### 5. Answer Completeness
5. Fully complete: The answer fully and comprehensively covers every aspect of the question.
4. Very complete: Answer addresses nearly all aspects of the question with appropriate depth and context.
3. Moderately complete: Answer covers most critical elements but lacks some details.
2. Partially complete: Answer addresses some key elements but omits several important aspects.
1. Severely incomplete: Answer is wrong or addresses only a minimal fraction of what was asked.
---
### EXAMPLES
{examples}
---
### CONTEXT FOR EVALUATION (Your turn)
**Question:**
{question}
**Answer:**
{answer}
---
### OUTPUT FORMAT
Your final output must contain be a single, valid JSON object. Do not include any text or explanations outside of the JSON structure.

{{
  "scientific_validity_question": {{"justification": "Brief justification for the score.","score": <integer_from_1_to_5>}},
  "scientific_validity_answer": {{"justification": "Brief justification for the score.","score": <integer_from_1_to_5>}},
  "question_complexity": {{"justification": "Brief justification for the score.","score": <integer_from_1_to_5>}},
  "specificity_answer": {{"justification": "Brief justification for the score.","score": <integer_from_1_to_5>}},
  "answer_completeness": {{"justification": "Brief justification for the score.","score": <integer_from_1_to_5>}}
}}