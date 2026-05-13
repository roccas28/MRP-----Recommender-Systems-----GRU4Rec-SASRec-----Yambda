# Sequential Recommender Systems: ETL & EDA Pipeline
**Course:** Major Research Project (MRP)
**Student:** Roccas Raveendran
**Student ID:** 500853856  
**Date:** May 6, 2026

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Project Overview
This module handles the Extract, Transform, and Load (ETL) pipeline and Exploratory Data Analysis (EDA) for the Yambda-50m dataset. It converts 50 million flat interaction logs into chronological, padded user sequences required for Transformer (SASRec) and RNN (GRU4Rec) architectures.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Project Structure
The codebase is modularized into six Python scripts and several designated directories:

### 1. `utils.py`
Handles memory-optimized data ingestion and label encoding.

### 2. `env.py`


### 3. `model.py`


### 4. `train.py`


### 5. `main.py`
The primary orchestrator. Cleans the data, applies the LOO split, and exports the final neural-network-ready arrays.

### 6. `plots.py`
Generates EDA visualizations (completion thresholds, long-tail popularity).

### 7. `data/`
**[IMPORTANT]** This folder is deliberately left empty in the submission zip file due to size constraints. The multi_event.parquet file must be downloaded via the link below and extracted here before running the code.
* **Download Link:** 

### 8. `saved_models/`


### 9. `output/`
Destination for the processed training/validation/testing matrices and EDA `.png` plots.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Installation & Requirements
This project requires **Python 3.10+**.

1. **Install Dependencies:**
   Ensure you have the necessary libraries installed by running:
   ```bash
   pip install -r requirements.txt