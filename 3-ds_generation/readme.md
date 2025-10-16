# 3\. Large-Scale Dataset Generation

This directory contains the scripts for the main, large-scale generation of the BioGraphletQA dataset. This process uses the best-performing prompt identified in the `2-prompt_ablation` study to generate the full set of raw question-answer pairs. It also includes a notebook for initial data analysis and filtering.

## File Descriptions

  * **`DS_GENERATION.py`**: The core Python script that performs the dataset generation. It takes the final graphlet file (from `1-kg_preprocessing`) as input and uses the selected prompt template to generate a QA pair for each graphlet.
  * **`3_gen.sh`**: An example Slurm batch script used to run `DS_GENERATION.py` on an HPC cluster. For our generation, this script was configured to process the graphlets in batches of approximately 30,000 for efficient and manageable execution.
  * **`data_analysis.ipynb`**: A Jupyter notebook for post-processing and analyzing the raw generated data. Its primary function is to perform Z-score filtering to remove outliers based on question and answer length.

## Workflow

1.  **Prerequisites**: Ensure you have the final, preprocessed graphlet file from the `1-kg_preprocessing` stage.
2.  **Run Generation**:
      * On an HPC cluster with Slurm, you can adapt and run the `3_gen.sh` script:
        ```bash
        sbatch 3_gen.sh
        ```
      * Alternatively, you can execute the Python script directly in your configured environment.
3.  **Post-Processing and Filtering**:
      * Once the generation process is complete, open and run the cells in the `data_analysis.ipynb` notebook. This will analyze the raw output and apply the Z-score filtering to produce the cleaned dataset.

## Z-Score Filtering

To ensure data quality and consistency, we removed outliers based on the length of the generated text. The `data_analysis.ipynb` notebook implements this filtering step by:

1.  Calculating the mean and standard deviation of the character length for all questions and answers.
2.  Discarding any QA pair where the length of either the question or the answer falls more than **three standard deviations** from the mean.

This step is crucial for removing anomalously short or long entries that are often indicative of low-quality or malformed generations.