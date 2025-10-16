Your role is to analyze the Question Answer pair, as well as the snippets provided.
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
