# Neural Architectures for Large-Scale Retrieval: Benchmarking Transformers against Recurrent Baselines
**Course:** Major Research Project (MRP)  
**Student:** Roccas Raveendran  
**Student ID:** 500853856  
**Date:** May 31, 2026

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Project Overview
This project performs a detailed empirical analysis of Transformer-based architectures versus Recurrent Neural Networks (RNNs) for sequential next-item prediction. Utilizing the 50-million interaction subset of the "Yambda" dataset (Yandex), an out-of-core ETL pipeline was engineered to process massive flat logs into chronological user sequences. The project compares **SASRec (Self-Attentive Sequential Recommendation)** against **GRU4Rec** using Bayesian Personalized Ranking (BPR) Loss and strict Negative Sampling. The models are evaluated using a Leave-One-Out temporal split on retrieval accuracy (Hit Rate @ 10) and ranking quality (NDCG @ 10).

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Project Structure
The codebase is modularized into eight Python scripts and three designated directories to separate data engineering, model architecture, and evaluation:

### 1. `utils.py`
Handles memory-optimized data ingestion and mapping out-of-core.
* Extracts statistical EDA samples without overloading RAM.
* Executes DuckDB SQL aggregations to group 50M flat rows into user arrays directly on the SSD.
* Generates the global label-encoding dictionary to dynamically map sparse items to contiguous integers.

### 2. `plots.py`
Generates all visualization artifacts for the final report.
* **EDA Plots:** Played ratio distributions, long-tail item popularity, event types, and sequence length variance.
* **Training Plots:** Comparative BPR loss curves across epochs for SASRec vs GRU4Rec.
* **Evaluation Plots:** Grouped bar charts visualizing final HR@10 and NDCG@10 metrics.

### 3. `main_etl.py`
The primary Data Engineering orchestrator.
* Triggers EDA metric extraction.
* Streams the DuckDB grouped data in discrete PyArrow batches.
* Applies the **Leave-One-Out (LOO) Temporal Split** (Train / Validation / Test).
* Exports the final padded, neural-network-ready arrays to `processed_sequences.parquet`.

### 4. `dataset.py`
Manages the TensorFlow `tf.data.Dataset` pipeline for GPU streaming.
* **Padding & Truncation:** Enforces a maximum context window of $K=50$, left-padding shorter sequences with zeroes.
* **Strict Negative Sampling:** Utilizes a custom Python generator and Hash Sets to perform $O(1)$ collision checks, ensuring generated negative items do not exist in the user's true history.

### 5. `model.py`
Defines the deep learning architectures using the `tf.keras.Model` API.
* **`SASRec`**: A Transformer architecture utilizing Multi-Head Attention, Positional Embeddings, and a **Causal Mask** to strictly enforce causal learning without future data leakage.
* **`GRU4Rec`**: The recurrent baseline utilizing update and reset gates to process the sequential arrays left-to-right.

### 6. `train.py`
Contains the custom training loops and mathematical gradient descent logic.
* **`RecommenderTrainer`**: Executes the forward pass to extract dense "mood vectors".
* Computes **Bayesian Personalized Ranking (BPR) Loss** via Sampled Softmax and updates network weights via `tf.GradientTape`.

### 7. `main_train.py`
The orchestrator for the Machine Learning phase.
* Dynamically calculates dataset vocabulary size.
* Builds and trains both SASRec and GRU4Rec sequentially for 10 epochs.
* Saves the final `.h5` model weights to disk and generates the comparative loss curves.

### 8. `evaluate.py`
Executes the strict 100-item ranking simulation for model grading.
* Rebuilds the models and loads the trained weights.
* Evaluates a user's target item against 99 random negative items using dot-product similarity.
* Calculates **Hit Rate (HR@10)**, **Normalized Discounted Cumulative Gain (NDCG@10)**, and compares them against a purely statistical **Popularity Baseline**.

---

### Directories

* **`Data/`**: **[IMPORTANT]** This folder is deliberately left empty in the submission zip file due to size constraints. The `flat_multievent_50m.parquet` file must be downloaded and extracted here before running the code.
* **`output/`**: **[IMPORTANT]** .parquet not included because of size constraints. Destination for the processed training/validation/testing matrices (`.csv` previews) and all generated `.png` plots.
* **`saved_models/`**: **[IMPORTANT]** This folder is deliberately left empty in the submission zip file due to size constraints. Checkpoint directory where the `.h5` tensor weights are saved post-training for evaluation loading.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Installation & Requirements
This project requires **Python 3.10+**.

1. **Install Dependencies:**
   Ensure you have the necessary data engineering and deep learning libraries installed by running:
   ```bash
   pip install -r requirements.txt