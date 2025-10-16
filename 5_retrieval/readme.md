# 5. Supporting Document Annotation

This directory contains the code responsible for the final data enrichment step: annotating retrieved PubMed documents to link them with our generated QA pairs.

## Overview

The goal of this stage is to augment the BioGraphletQA dataset by grounding each question-answer pair in unstructured text from the scientific literature. This process uses a Large Language Model (LLM) to act as an annotator, evaluating the relevance of pre-retrieved documents and extracting supporting text snippets.

**Note:** The initial retrieval code, which used **BM25** to search a local PubMed index, is not included in this repository. Distributing the large and complex search indexes required for this step is not feasible. This directory focuses exclusively on the subsequent LLM-based annotation part of the pipeline.

## Workflow & File Description

There is a single script in this directory that handles the entire LLM annotation process.

* **`llm_annotate_retrieved_docs.py`** (or a similar name): This script takes the generated QA pairs and their top 10 most relevant documents (retrieved via BM25) as input. For each document, it prompts an LLM to perform two tasks:
    1.  **Classify Relevance**: Determine if the document is truly relevant to the given QA pair.
    2.  **Extract Snippet**: If the document is relevant, extract the most pertinent sentence or short passage that supports the facts in the QA pair.

Due to the large number of API calls required (1 QA pair × 10 documents = 10 prompts), the script is designed to be run in batches of **5,000 QA pairs** for efficient and manageable processing. The final output is a collection of QA–document–snippet triples that enrich the final dataset.