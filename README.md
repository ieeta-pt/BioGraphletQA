# BioGraphletQA: A Framework for Complex KGQA Dataset Generation

**Anonymous ARR Submission**  
**Title:** *From Graphs to Questions: A Framework for Complex KGQA Dataset Generation*

---

## Note
This is a work-in-progress repository submitted for review. All code provided here is functional and has been anonymized to the best of our ability. We kindly ask reviewers to refrain from seeking identifying information that may not have been fully anonymized. Note that a large portion of the data and outputs (typically found in directories beginning with an underscore, `_`) are currently omitted and will be made available via Zenodo after the review process.

---

## Overview

**BioGraphletQA** is a large-scale dataset and pipeline designed for generating complex **Knowledge Graph Question Answering (KGQA)** datasets using structured methodologies and **Large Language Models (LLMs)**.

At the core of our approach is the use of **graphlets**—small subgraphs extracted from the **OREGANO knowledge graph**—to anchor the generation of diverse, high-quality QA pairs.

---

## Pipeline

The dataset generation process consists of **five main stages**:
1.  **Graph Hydration & Reduction**
    *   We preprocess the OREGANO KG to extract a relevant, manageable subset.
    *   Resulting graphlets are small, self-contained, and optimized for QA generation.

2.  **Prompt Engineering & Ablation Study**
    *   We conduct a comprehensive prompt ablation study to determine the most effective prompt for generating complex, high-quality QA pairs using LLMs.

3.  **LLM-Based Filtering**
    *   A second LLM pass filters out low-quality QA pairs.
    *   Ensures that only scientifically valid and complete QA pairs are retained, a process validated by human expert evaluation.

4.  **Supporting Document Retrieval**
    *   A subset of 50,000 QA pairs is enriched with supporting documents from PubMed.
    *   We use BM25 for initial retrieval, followed by an LLM to confirm relevance and extract precise evidence snippets, creating QA–document–snippet triples.

5.  **Task-Specific Rephrasing**
    *   We rephrase a subset of QA pairs to match the formats of downstream benchmarks like PubMedQA (yes/no) and MedQA (multiple-choice).
    *   This alignment facilitates standardized evaluation and allows for robust assessment of model performance on established tasks.

## Repository Structure

In the repository usually scripts/notebooks are numbered in the order they should be run. 

```text
├── 0-kg_aquisition/          # Script for downloading the OREGANO KG
├── 1-kg_processing/          # KG hydration,reduction and graphlet extraction logic
├── 2-prompt_ablation/        # Prompt variants and ablation experiment setup
├── 3-ds_generation/          # LLM-based question-answer pair generation
├── 4-ds_filtering/           # LLM-based quality filtering and human annotation
├── 5-retrieval/              # LLM-based retrieval annotation, does not include code for BM25 retrieval
├── 6-rephrasing/             # Downstream task specific rephrasing: MedQA and PubMedQA 

│
├── _figures/                 # Paper figures and visualizations
├── _generation_templates/    # Actual graphlets used in QA generation
├── _graph/                   # KG data
├── _llm_outs/                # Any output from an LLM
├── _prompts/                 # KG data

```

---
## Requirements

This project uses two environments:

- **Graph Tool** (version 2.92, commit `fa53df3d`) must be installed via Conda due to system dependencies:

  ```bash
    conda create -n graph-tool-env -c conda-forge graph-tool=2.92
    conda activate graph-tool-env
  ```

- **Main environment** for all other Python packages, installed with:
  
  ```bash
    pip install -r requirements.txt
  ```


## Dataset Statistics

- **Total QA Pairs:** 119,856  
- **Graphlet Size:** 3 to 5 nodes per graphlet  
- **QA Format:** Each question is tightly linked to its corresponding graphlet context



| **ID** | **Total** | **Downsampling** | | **Generated** | **Acceptance** | |
| :--- | ---: | :--- | ---: | ---: | :--- | ---: |
| | | **Ratio** | **Count** | | **Total** | **Ratio** |
|---|---|---|---|---|---|---|
| 1 | 2,980,635 | $3.35 \times 10^{-3}$ | 9,954 | 9,913 | 4,544 | 45.8 % |
| 2 | 3,702 | $1.00 \times 10^{+00}$ | 3,702 | 3,690 | 1,744 | 47.3 % |
|---|---|---|---|---|---|---|
| 3 | 50,513,861 | $1.98 \times 10^{-4}$ | 9,826 | 9,783 | 4,149 | 42.4 % |
| 4 | 41,964,954 | $2.38 \times 10^{-4}$ | 10,108 | 10,021 | 4,475 | 44.7 % |
| 5 | 3,609,661 | $2.77 \times 10^{-3}$ | 10,165 | 10,103 | 5,325 | 52.7 % |
| 6 | 71,664 | $1.40 \times 10^{-1}$ | 9,913 | 9,810 | 4,485 | 45.7 % |
| 7 | 13,537 | $7.39 \times 10^{-1}$ | 9,939 | 9,870 | 4,365 | 44.2 % |
| 8 | 11,794 | $8.48 \times 10^{-1}$ | 10,038 | 9,948 | 5,212 | 52.4 % |
|---|---|---|---|---|---|---|
| 9 | 1,080,297,928 | $9.26 \times 10^{-6}$ | 9,988 | 9,817 | 3,485 | 35.5 % |
| 10 | 1,810,874,588 | $5.52 \times 10^{-6}$ | 9,952 | 9,806 | 3,679 | 37.5 % |
| 11 | 584,613,716 | $1.71 \times 10^{-5}$ | 10,126 | 9,939 | 4,390 | 44.2 % |
| 12 | 922,997 | $1.08 \times 10^{-2}$ | 10,078 | 9,874 | 4,144 | 42.0 % |
| 13 | 772,905 | $1.29 \times 10^{-2}$ | 10,100 | 9,885 | 3,897 | 39.4 % |
| 14 | 871,384 | $1.15 \times 10^{-2}$ | 9,946 | 9,723 | 4,275 | 44.0 % |
| 15 | 46,904 | $2.13 \times 10^{-1}$ | 10,001 | 9,823 | 3,628 | 36.9 % |
| 16 | 166,337,860 | $6.01 \times 10^{-5}$ | 9,946 | 9,841 | 5,087 | 51.7 % |
| 17 | 239,193 | $4.18 \times 10^{-2}$ | 10,143 | 9,949 | 4,459 | 44.8 % |
| 18 | 6,267 | $1.00 \times 10^{+00}$ | 6,267 | 6,103 | 2,725 | 44.7 % |
| 19 | 225,464 | $4.44 \times 10^{-2}$ | 10,088 | 9,878 | 3,606 | 36.5 % |
| 20 | 74,698,349 | $1.34 \times 10^{-4}$ | 9,894 | 9,781 | 5,496 | 56.2 % |
| 21 | 55,278 | $1.81 \times 10^{-1}$ | 9,878 | 9,629 | 3,281 | 34.1 % |
| 22 | 79,900 | $1.25 \times 10^{-1}$ | 10,013 | 9,846 | 4,533 | 46.0 % |
| 23 | 65,548 | $1.53 \times 10^{-1}$ | 10,031 | 9,621 | 4,629 | 48.1 % |
| 24 | 11,395 | $8.78 \times 10^{-1}$ | 9,976 | 9,741 | 4,149 | 42.6 % |
| 25 | 31,145 | $3.21 \times 10^{-1}$ | 9,989 | 9,781 | 3,759 | 38.4 % |
| 26 | 5,617 | $1.00 \times 10^{+00}$ | 5,617 | 5,292 | 3,036 | 57.4 % |
| 27 | 3,810 | $1.00 \times 10^{+00}$ | 3,810 | 3,690 | 1,647 | 44.6 % |
| 28 | 18,217 | $5.49 \times 10^{-1}$ | 10,067 | 9,577 | 5,593 | 58.4 % |
| 29 | 44,022 | $2.27 \times 10^{-1}$ | 10,019 | 9,639 | 6,059 | 62.9 % |

---

## Quality Assessment

To validate dataset quality:

- A **domain expert** manually annotated **53 QA pairs** using five quality criteria:
  - Scientific Validity  
  - Question Complexity  
  - Answer Completeness  
  - Semantic Coherence  
  - Answer Specificity (identified as a future improvement area)

---

