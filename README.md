# Medical Lab Result SLM Fine-Tuning

This repository contains a synthetic medical lab dataset and a Jupyter Notebook pipeline designed to fine-tune Small Language Models (SLMs) like DeepSeek-R1-Distill-Llama-8B to act in the role of a highly capable medical laboratory assistant. 

The primary goal of this project is to build an AI assistant that can analyze common lab results, correctly identify abnormalities based on standard clinical reference ranges, and intelligently recommend logical follow-up diagnostic tests.

## 📂 Repository Contents

*   **`fine_tuning_lab_tests.jsonl`**: The primary dataset formatted in JSONL (JSON Lines) ready for conversational fine-tuning.
*   **`fine_tuning_lab_tests_readable.json`**: A formatted, pretty-printed version of the dataset meant for human review and analysis.
*   **`fine_tune_unsloth.ipynb`**: A complete Jupyter Notebook pipeline for fine-tuning the model locally. It is heavily optimized (using Unsloth and 4-bit LoRA) to run comfortably on an 8GB VRAM GPU.

## 📊 Dataset Overview

The dataset consists of **100 conversational records** spread evenly across **10 common and vital laboratory tests**:

1.  Glucose (Fasting & Random)
2.  Hemoglobin (Hgb)
3.  Creatinine (Serum)
4.  Sodium (Serum)
5.  Potassium (Serum)
6.  Cholesterol (Total)
7.  White Blood Cell (WBC) Count
8.  Platelets
9.  Alanine Aminotransferase (ALT)
10. Calcium (Serum)

### Structure of the Data
Each record is structured as a chat completion with exactly three roles:
*   **System**: Instructs the model on its persona ("You are a medical laboratory assistant...").
*   **User**: Provides the Test Name, the specific Result, and the normal Reference Range. *(Note: LOINC codes have been intentionally excluded from the input to promote natural language processing).*
*   **Assistant**: Contains the AI's analysis, stating whether the result is Normal, Borderline, High, Low, or Critical. Crucially, it also includes **diagnostic follow-up suggestions** (e.g., suggesting an HbA1c test for a high glucose reading).

## 🚀 Fine-Tuning Environment (8GB VRAM Optimized)

The `fine_tune_unsloth.ipynb` notebook provides a step-by-step guide to fine-tuning. It utilizes the [Unsloth](https://github.com/unslothai/unsloth) library, which is currently the gold standard for memory-efficient LLM/SLM fine-tuning on consumer hardware.

**Optimizations Included:**
*   **4-Bit Quantization**: Shrinks the 8 Billion parameter model down to ~5.5GB of VRAM.
*   **LoRA Adapters (Rank 16)**: Rather than full fine-tuning, tiny adapter weights are trained on the attention and MLP layers.
*   **Gradient Checkpointing & Accumulation**: Keeps memory spikes low during the backward pass while simulating larger batch sizes.
*   **DeepSeek-R1 CoT Formatting**: The dataset is automatically mapped into the specific `<think>` tags required by the DeepSeek-R1 reasoning models to maintain chain-of-thought capabilities.

### Prerequisites

To run the notebook locally, you will need:
*   An NVIDIA GPU with at least 8GB of VRAM.
*   A Hugging Face account and Access Token (for downloading the base model).
*   (Optional) A Weights & Biases account (for tracking training loss).

## 💡 Usage Instructions

1.  Clone this repository to your local machine:
    ```bash
    git clone https://github.com/israeltn/medical_lab_result_SLM.git
    cd medical_lab_result_SLM
    ```
2.  Open `fine_tune_unsloth.ipynb` in your preferred Jupyter environment (e.g., JupyterLab, VS Code).
3.  Set your Hugging Face API token in the authentication block.
4.  Run all cells sequentially. 
5.  *(Optional)* If you hit an Out Of Memory (OOM) error, try reducing the `max_seq_length` variable from `2048` to `1024` or `512` in Step 2 of the notebook.

## ⚠️ Medical Disclaimer
This dataset is **synthetic** and generated for the purpose of AI training and research. It should **not** be used for actual medical diagnosis without the oversight of a licensed clinical professional.
