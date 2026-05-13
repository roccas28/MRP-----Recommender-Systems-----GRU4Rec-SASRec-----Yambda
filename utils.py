# Roccas Raveendran - 500853856
# Sequential Recommender System - Utility & Out-of-Core ETL Functions

import pandas as pd
import numpy as np
import duckdb
import os
import pyarrow.parquet as pq

DATA_PATH = './Data/' # Define root data directory -> Location of downloaded Parquet files

def extract_eda_sample(file_name='multi_event.parquet', sample_percent=10): # Extracts random sample -> Prevents memory overflow during EDA
    print(f"Extracting {sample_percent}% random sample for EDA plotting...")
    file_path = os.path.join(DATA_PATH, file_name) # Construct full file path -> Locates Parquet on SSD
    
    # Execute SQL query to randomly sample data -> Hyperparameter: 10% balances RAM safety with statistical validity
    query = f"""
        SELECT played_ratio_pct, item_id 
        FROM '{file_path}' 
        USING SAMPLE {sample_percent}%
    """
    return duckdb.query(query).df() # Convert DuckDB relation to Pandas DataFrame -> Returns sample

def duckdb_group_and_export(file_name='multi_event.parquet', out_file='./output/raw_grouped.parquet'): # Groups raw logs into user sequences out-of-core
    print("Using DuckDB to group 50M rows out-of-core and export directly to disk...")
    file_path = os.path.join(DATA_PATH, file_name)
    
    # Bypasses RAM entirely by writing the SQL output directly to a Parquet file -> Prevents crash on 50M rows
    query = f"""
        COPY (
            SELECT 
                uid,
                list(item_id ORDER BY timestamp ASC) as item_sequence
            FROM '{file_path}'
            WHERE uid IS NOT NULL 
              AND item_id IS NOT NULL 
              AND timestamp IS NOT NULL
            GROUP BY uid
            HAVING count(item_id) >= 5 
        ) TO '{out_file}' (FORMAT PARQUET);
    """
    # Hyperparameter Note: HAVING count >= 5 filters sparsity -> Mitigates cold-start problem & guarantees enough data for LOO split
    duckdb.execute(query) # Execute query directly to disk -> Creates temporary grouped parquet
    print("DuckDB aggregation successfully saved to disk!")

def get_global_item_mapping(file_name='multi_event.parquet'): # Generates master ID mapping dictionary
    print("Extracting global unique items for Label Encoding...")
    file_path = os.path.join(DATA_PATH, file_name)
    
    # Get unique items via SQL -> Super fast distinct extraction with very low RAM overhead
    unique_items_df = duckdb.query(f"SELECT DISTINCT item_id FROM '{file_path}' WHERE item_id IS NOT NULL").df()
    item_ids = unique_items_df['item_id'].values # Extract as numpy array -> Prepares for dictionary comprehension
    
    # Create dictionary mapping -> Maps original sparse IDs to contiguous integers starting at 1 (0 is reserved for padding)
    item2idx = {int(item): idx + 1 for idx, item in enumerate(item_ids)}
    
    return item2idx # Returns mapping dictionary

def extract_full_eda_stats(file_name='multi_event.parquet'): # Aggregates exact counts for all 50M rows out-of-core
    print("Extracting full sequence lengths and event distributions via DuckDB...")
    file_path = os.path.join(DATA_PATH, file_name)
    
    # 1. Get Sequence Lengths (100% of data) -> Extracts total interaction count per user
    query_lengths = f"""
        SELECT count(item_id) as seq_length 
        FROM '{file_path}' 
        GROUP BY uid 
        HAVING count(item_id) >= 5
    """
    df_lengths = duckdb.query(query_lengths).df() # Execute and convert to DataFrame
    
    # 2. Get Event Type Distribution (100% of data) -> Extracts total counts for likes, listens, skips, etc.
    query_events = f"""
        SELECT event_type, count(*) as total_count 
        FROM '{file_path}' 
        GROUP BY event_type
    """
    df_events = duckdb.query(query_events).df() # Execute and convert to DataFrame
    
    return df_lengths, df_events # Return both statistical dataframes

def export_processed_preview(file_path='./output/processed_sequences.parquet', out_path='./output/processed_preview.csv'): # Creates visual subset for reporting
    print(f"Exporting a 15-row preview of the final arrays to {out_path}...")
    
    # Read only the first 15 rows -> Hyperparameter: 15 is enough to visualize tensor structure without loading whole file
    pf = pq.ParquetFile(file_path) # Open file mapping -> Does not load into RAM
    preview_df = next(pf.iter_batches(batch_size=15)).to_pandas() # Extract single small batch -> Converts to Pandas
    preview_df.to_csv(out_path, index=False) # Save as CSV -> Formats for Overleaf document embedding
