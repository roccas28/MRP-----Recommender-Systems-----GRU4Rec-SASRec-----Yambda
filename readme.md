# Neural Architectures for Large-Scale Retrieval: Benchmarking Transformers against Recurrent Baselines
**Course:** Major Research Paper(MRP)  
**Student:** Roccas Raveendran  
**Student ID:** 500853856  

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Project Overview
This project performs a detailed empirical analysis of Transformer-based architectures versus Recurrent Neural Networks (RNNs) for sequential next-item prediction. Utilizing the 50-million interaction subset of the "Yambda" dataset (Yandex), an out-of-core ETL pipeline was engineered to process massive flat logs into chronological user sequences. The project strictly evaluates**SASRec (Self-Attentive Sequential Recommendation)** against **GRU4Rec** using Bayesian Personalized Ranking (BPR) Loss and strict Negative Sampling. To eliminate data leakage and ensure temporal validity, the models are trained using a Leave-One-Out (LOO) temporal split. An automated, 3D grid search optimization phase maps architectural sensitivities before final training. Models are graded on retrieval accuracy (Hit Rate @ 10) and ranking quality (NDCG @ 10) against a competitive statistical Popularity baseline.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Project Structure
The codebase is modularized into nine Python scripts and three designated directories to separate data engineering, hyperparameter optimization, model architecture, and final evaluation:

### 1. `utils.py`
Handles memory-optimized data ingestion and mapping out-of-core.
* Extracts statistical EDA samples without overloading RAM.
* Executes DuckDB SQL aggregations to group 50M flat rows into user arrays directly on the SSD.
* Generates the global label-encoding dictionary to dynamically map sparse items to contiguous integers.

### 2. `plots.py`
Generates all visualization artifacts for the final report.
* **EDA Plots:** Played ratio distributions, long-tail item popularity, event types, and sequence length variance.
* **Grid Search Plots:** Generates isolated 1x3 line graphs capturing independent hyperparameter scaling laws and dual multidimensional heatmaps capturing interaction effects between sequence length and embedding dimensionality.
* **Training Plots:** Comparative BPR loss curves across epochs for SASRec vs GRU4Rec.
* **Evaluation Plots:** Grouped bar charts visualizing final HR@10 and NDCG@10 metrics.

### 3. `main_etl.py`
The primary Data Engineering orchestrator.
* Triggers EDA metric extraction.
* Streams the DuckDB grouped data in discrete PyArrow batches.
* Applies the **Leave-One-Out (LOO) Temporal Split** (Train / Validation / Test).
* Exports the final padded, neural-network-ready arrays to `processed_sequences.parquet`using an optimized context window of $K=70$.

### 4. `dataset.py`
Manages the TensorFlow `tf.data.Dataset` pipeline for GPU streaming.
* **Padding & Truncation:** Enforces variable context windows(up to $K=100$ ), dynamically during optimization, left-padding shorter sequences with zeroes.
* **Strict Negative Sampling:** Utilizes a custom Python generator and Hash Sets to perform $O(1)$ collision checks, ensuring generated negative items do not exist in the user's true history.

### 5. `model.py`
Defines the deep learning architectures using the `tf.keras.Model` API.
* **`SASRec`**: A Transformer architecture utilizing Multi-Head Attention, Positional Embeddings, and a **Causal Mask** to strictly enforce causal learning without future data leakage.
* **`GRU4Rec`**: The recurrent baseline utilizing update and reset gates to process the sequential arrays left-to-right.

### 6. `train.py`
Contains the custom training loops and mathematical gradient descent logic.
* **`RecommenderTrainer`**: Executes the forward pass to extract dense "mood vectors". It accepts an optional optimizer parameter to accommodate custom learning rates passed dynamically during grid tuning while maintaining backward compatibility with standard training protocols.
* Computes **Bayesian Personalized Ranking (BPR) Loss** via Sampled Softmax and updates network weights via `tf.GradientTape`.

### 7. `tune.py`
The automated 3D Grid Search Orchestrator.
* Systematically trains and evaluates 150 unique neural network configurations across both architectures to prevent manual or heuristic tuning biases.
* Tests explicit grid permutations across Maximum Sequence Length (K in [10, 30, 50, 70, 100]), Embedding Dimensionality (d in [16, 32, 64, 128, 256]), and Learning Rates (0.0005, 0.001, 0.005).
* Leverages explicit calls to 'tf.keras.backend.clear_session()' between iterative runs to clear Video RAM (VRAM) and prevent memory overflow exceptions during continuous training loops.
* Exports comprehensive metrics to 'comprehensive_grid_search.csv' and triggers the new visualization methods in 'plots.py'.

### 8. `main_train.py`
The orchestrator for the Machine Learning phase.
* Ingests the optimal structural constraints established during the hyperparameter tuning phase ($K=70$, $Learning Rate = 0.005$).
* Configures SASRec for a low-capacity build ($d=32$) and GRU4Rec for a high-capacity build ($d=256$) to prevent early-epoch transformer overfitting while maximizing recurrent pattern memorization.
* Builds and trains both SASRec and GRU4Rec sequentially for 10 epochs.
* Saves the final `.h5` model weights to disk and generates the comparative training loss visual summaries.

### 9. `evaluate.py`
Executes the strict 100-item ranking simulation for model grading.
* Rebuilds the models according to the optimized structural configurations and loads the trained weights.
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

## Execution Workflow
   To reproduce the entire pipeline from scratch, execute the following order:
   ```bash
   python main_etl.py     # Step 1: Run Data Engineering & Preprocessing
   python tune.py         # Step 2: Run 150-Configuration Grid Search
   python main_train.py   # Step 3: Train Optimized Models for 10 Epochs
   python evaluate.py     # Step 4: Run Ranking Simulations & Final Evaluation