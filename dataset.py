# Roccas Raveendran - 500853856
# Sequential Recommender System - TensorFlow Dataset & Strict Negative Sampling

import pandas as pd
import numpy as np
import tensorflow as tf
import pyarrow.parquet as pq

class SequentialDataLoader: # Manages data ingestion, padding, and STRICT negative sampling
    
    def __init__(self, file_path='./output/processed_sequences.parquet', max_len=50):
        print("Initializing Strict Sequential DataLoader...")
        self.file_path = file_path
        self.max_len = max_len 
        
        # Load the preprocessed LOO arrays into memory
        df = pd.read_parquet(self.file_path, engine='pyarrow')
        
        # Convert Pandas columns to native Python lists/arrays  -> Massively accelerates iteration speed during generation
        self.uids = df['uid'].values
        self.train_seqs = df['train_sequence'].tolist()
        self.val_items = df['val_item'].values
        self.test_items = df['test_item'].values
        
        # FIX: Dynamically calculate the maximum Item ID directly from the dataset!
        # This guarantees the Negative Sampler and the Embedding layers perfectly match the data.
        print("Scanning arrays to determine dataset cardinality...")
        flat_train = np.concatenate(self.train_seqs) # Flattens all history arrays into one massive 1D array
        
        # Find the absolute highest ID across the Train, Val, and Test splits
        self.total_items = int(max(flat_train.max(), self.val_items.max(), self.test_items.max()))
        print(f"Dynamically set total unique items to: {self.total_items}")
        
    def _pad_and_truncate(self, seq): # Enforces a strict uniform tensor shape

        # FIX: Force the PyArrow/NumPy object to be a flat native Python list of integers.
        # This completely prevents nested/inhomogeneous arrays during padding.
        seq = [int(x) for x in seq]
        
        if len(seq) > self.max_len:
            # Truncate -> Keep only the most recent K interactions
            return seq[-self.max_len:]
        else:
            # Pad -> Add 0s to the LEFT side so the most recent items align perfectly on the right
            pad_length = self.max_len - len(seq)
            return [0] * pad_length + seq

    def _data_generator(self, is_training): # Pure Python generator -> Bypasses TF Graph limitations for dynamic loops
        
        # Create an array of indices corresponding to our users
        indices = np.arange(len(self.train_seqs))
        
        # If training, shuffle the indices every single epoch -> Ensures the network doesn't memorize user order
        if is_training:
            np.random.shuffle(indices)
            
        for idx in indices:
            seq = self.train_seqs[idx]
            
            # Determine the positive target based on the phase
            pos_target = self.val_items[idx] if is_training else self.test_items[idx]
            
            # --- STRICT NEGATIVE SAMPLING ---
            # 1. Create a Hash Set of all items the user has EVER interacted with
            invalid_items = set(seq)
            invalid_items.add(self.val_items[idx])
            invalid_items.add(self.test_items[idx])
            
            # 2. Draw a random integer
            neg_target = np.random.randint(1, self.total_items + 1)
            
            # 3. Collision Check -> While the random number is in the Hash Set, draw a new one
            # Because we use a Set, this check takes O(1) time and prevents False Negatives from polluting the gradient
            while neg_target in invalid_items:
                neg_target = np.random.randint(1, self.total_items + 1)
                
            padded_seq = self._pad_and_truncate(seq)
            
            # Yield the final dictionary structure required by the Keras models
            yield (
                {"input_sequence": np.array(padded_seq, dtype=np.int32)},
                {"positive_target": np.int32(pos_target), "negative_target": np.int32(neg_target)}
            )

    def generate_tf_dataset(self, batch_size=256, is_training=True): # Constructs the tf.data pipeline
        
        print(f"Constructing tf.data.Dataset (Training={is_training})...")
        
        # Tell TensorFlow exactly what shape and datatype to expect from our Python generator
        output_signature = (
            {
                "input_sequence": tf.TensorSpec(shape=(self.max_len,), dtype=tf.int32)
            },
            {
                "positive_target": tf.TensorSpec(shape=(), dtype=tf.int32),
                "negative_target": tf.TensorSpec(shape=(), dtype=tf.int32)
            }
        )
        
        # Wrap the Python generator in the high-performance tf.data API
        dataset = tf.data.Dataset.from_generator(
            lambda: self._data_generator(is_training=is_training),
            output_signature=output_signature
        )
        
        # Batch and prefetch -> Prefetching allows the CPU to generate the next batch while the GPU trains on the current one
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
        return dataset
