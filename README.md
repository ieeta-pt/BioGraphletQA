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

The dataset generation process consists of **three main stages**:

1. **Graph Hydration & Reduction**  
   - We preprocess the OREGANO KG to extract a relevant, manageable subset.
   - Resulting graphlets are small, self-contained, and optimized for QA generation.

2. **Prompt Engineering & Ablation Study**  
   - We conduct a comprehensive prompt ablation study to determine the most effective prompt for generating complex, high-quality QA pairs using LLMs.

3. **LLM-Based Filtering**  
   - A second LLM pass filters out low-quality QA pairs.
   - Ensures that only scientifically valid and complete QA pairs are retained.

---


## Repository Structure

In the repository usually scripts/notebooks are numbered in the order they should be run. 

```text
├── 0-kg_aquisition/          # Script for downloading the OREGANO KG
├── 1-kg_processing/          # KG hydration,reduction and graphlet extraction logic
├── 2-prompt_ablation/        # Prompt variants and ablation experiment setup
├── 3-ds_generation/          # LLM-based question-answer pair generation
├── 4-ds_filtering/           # LLM-based quality filtering and human annotation
│
├── _figures/                 # Paper figures and visualizations
├── _generation_templates/    # Actual graphlets used in QA generation
├── _graph/                   # KG data
├── _llm_outs/                # Any output from an LLM

```

---
## Requirements

This project uses two environments:

- **Graph Tool** (version 2.92, commit `fa53df3d`) must be installed via Conda due to system dependencies:

  ```bash
    conda create -n graph-tool-env -c conda-forge graph-tool=2.92
    conda activate graph-tool-env```

- **Main environment** for all other Python packages, installed with:
  
  ```bash
    pip install -r requirements.txt```


## Dataset Statistics

- **Total QA Pairs:** 119,856  
- **Graphlet Size:** 3 to 5 nodes per graphlet  
- **QA Format:** Each question is tightly linked to its corresponding graphlet context

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

