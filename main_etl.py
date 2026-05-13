# Roccas Raveendran - 500853856
# Sequential Recommender System - Main ETL Pipeline Orchestrator

import pandas as pd
import numpy as np
import time
import os
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Import Custom Modules
import utils as ut
import plots as pt

def run():
    print("MRP Project: Sequential Recommender ETL & EDA Pipeline")
    
    # Ensure output directory exists -> Prevents FileNotFoundError on first run
    if not os.path.exists('./output'):
        os.makedirs('./output')

    start_time = time.time() # Initialize timer -> Tracks total ETL execution time
    
    # --- PHASE 1: Exploratory Data Analysis ---
    print("\n--- Running Exploratory Data Analysis ---")
    eda_df = ut.extract_eda_sample('multi_event.parquet', sample_percent=10) # Extract 10% sample
    pt.plot_played_ratio_distribution(eda_df) # Plot implicit threshold logic
    pt.plot_long_tail_popularity(eda_df) # Plot sampling justification
    del eda_df # Delete dataframe -> Immediately frees RAM for next operations

    # Generate full dataset EDA metrics
    df_lengths, df_events = ut.extract_full_eda_stats('multi_event.parquet') # Query out-of-core counts
    pt.plot_sequence_length_distribution(df_lengths) # Plot sequence truncations logic
    pt.plot_event_type_distribution(df_events) # Plot event composition
    del df_lengths, df_events # Delete dataframes -> Frees RAM
    
    # --- PHASE 2: Heavy ETL Grouping ---
    print("\n--- Running Data Preprocessing ---")
    raw_grouped_path = './output/raw_grouped.parquet' # Define temporary disk path
    ut.duckdb_group_and_export('multi_event.parquet', raw_grouped_path) # Group 50M flat rows into chronological arrays
    
    # Extract Label Encoding Dictionary
    item2idx = ut.get_global_item_mapping('multi_event.parquet') # Get contiguous ID map
    print(f"Total Unique Items Found: {len(item2idx)}")
    
    # --- PHASE 3: Streaming, Encoding & LOO Split ---
    print("\n--- Executing Label Encoding and LOO Temporal Split ---")
    
    writer = None # Initialize Parquet writer
    uid_counter = 1 # Initialize continuous UID counter
    
    # Context manager 'with open' -> Forces Windows to release the file lock after execution completes
    with open(raw_grouped_path, 'rb') as f:
        parquet_file = pq.ParquetFile(f)
        
        # Hyperparameter: batch_size=100,000 -> Streams 100k users at a time. Optimizes RAM footprint vs I/O speed
        for batch in tqdm(parquet_file.iter_batches(batch_size=100000), desc="Processing Batches"):
            df_chunk = batch.to_pandas() # Convert PyArrow batch to Pandas DataFrame
            
            # Sequentially encode user IDs -> Creates contiguous user mapping starting at 1
            df_chunk['mapped_uid'] = range(uid_counter, uid_counter + len(df_chunk))
            uid_counter += len(df_chunk) # Increment counter for next batch
            
            train_seqs, val_seqs, test_seqs = [], [], [] # Initialize empty arrays for LOO split
            
            for seq in df_chunk['item_sequence']: # Iterate through each user's chronological history
                
                mapped_seq = [item2idx[x] for x in seq] # Apply item mapping dict -> Replaces raw IDs with encoded integers
                
                # Execute Leave-One-Out (LOO) Temporal Split via array indexing
                train_seqs.append(mapped_seq[:-2]) # Appends all items EXCEPT the last two -> Input sequence
                val_seqs.append(mapped_seq[-2])    # Appends the second-to-last item -> Hyperparameter Tuning Target
                test_seqs.append(mapped_seq[-1])   # Appends the absolute last item -> Final Evaluation Target
                
            # Construct final subset tensor structure
            final_chunk = pd.DataFrame({
                'uid': df_chunk['mapped_uid'].astype(np.int32),
                'train_sequence': train_seqs,
                'val_item': np.array(val_seqs, dtype=np.int32),
                'test_item': np.array(test_seqs, dtype=np.int32)
            })
            
            # Convert chunk back to PyArrow Table -> Prepares for disk streaming
            table = pa.Table.from_pandas(final_chunk)
            
            # Initialize writer on first pass -> Uses table schema to define Parquet structure
            if writer is None:
                writer = pq.ParquetWriter('./output/processed_sequences.parquet', table.schema)
            writer.write_table(table) # Stream final processed batch to disk
            
    if writer:
        writer.close() # Close writer -> Secures final Parquet file
        
    os.remove(raw_grouped_path) # Delete temporary file -> Frees SSD storage since Windows lock is released

    # Generate the CSV preview -> Used for Table 1 in MRP Methodology Document
    ut.export_processed_preview()
    
    print(f"Successfully exported final padded sequences to ./output/processed_sequences.parquet")
    print(f"\nETL Pipeline Complete! Time elapsed: {(time.time() - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    run() # Execute main block
