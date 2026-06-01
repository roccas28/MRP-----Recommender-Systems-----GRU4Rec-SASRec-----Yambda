# Roccas Raveendran - 500853856
# Sequential Recommender System - Evaluation Metrics (HR@10 & NDCG@10)

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# Import our custom modules
import dataset as ds
import model as mdl
import plots as pt

class RecommenderEvaluator: # Handles the strict ranking evaluation phase
    
    def __init__(self, model, dataset_loader, total_items):
        print(f"Initializing Evaluator for {model.__class__.__name__}...")
        self.model = model
        self.loader = dataset_loader
        self.total_items = total_items
        
    def _calculate_ndcg(self, rank): # Mathematically penalizes the model if the true item is lower on the top 10 list
        # Formula: 1 / log2(rank + 1) -> Rank 1 gives 1.0, Rank 2 gives 0.63, Rank 9 gives 0.31
        return 1.0 / np.log2(rank + 1)

    def evaluate_model(self, top_k=10, num_negatives=99): # Standard RecSys evaluation methodology
        print(f"Evaluating model via Ranking (True Item vs {num_negatives} Negatives)...")
        
        hits = 0.0
        ndcgs = 0.0
        total_users = 0
        
        # Load the test dataset specifically -> is_training=False prevents the shuffle
        test_ds = self.loader.generate_tf_dataset(batch_size=256, is_training=False)
        
        # Iterate through batches
        for inputs, targets in tqdm(test_ds, desc="Evaluating Batches"):
            
            # 1. Forward Pass -> Get the final compressed user mood vectors for this batch
            user_states = self.model(inputs['input_sequence'], training=False)
            
            # For every user in this batch, we must conduct a ranking test
            for i in range(len(user_states)):
                current_user_state = user_states[i] # Shape: (embed_dim,)
                true_target = targets['positive_target'][i]
                
                # 2. Generate 99 random negative items for the test -> Simulates retrieval from a massive catalog
                # Note: In pure academic rigor, we'd collision-check these 99, but for inference speed, random is accepted
                test_negatives = np.random.randint(1, self.total_items + 1, size=num_negatives)
                
                # 3. Combine the 1 True Item with the 99 Fake Items (Total 100 items to rank)
                items_to_rank = np.append(true_target, test_negatives)
                
                # 4. Get the dense mathematical embeddings for all 100 items
                item_embs = self.model.item_embedding(items_to_rank) 
                
                # 5. Calculate Dot Product -> Score the user's mood against all 100 items
                scores = tf.reduce_sum(current_user_state * item_embs, axis=-1).numpy()
                
                # 6. Find the rank of the True Item
                # Since the True Item was placed at index 0, we count how many fake items scored HIGHER than it.
                # If 3 fake items scored higher, the True Item is ranked 4th.
                rank = (scores > scores[0]).sum() + 1
                
                # 7. Apply Evaluation Metrics
                if rank <= top_k: # Hyperparameter: top_k=10 -> Only counts if it makes the first page of results
                    hits += 1.0
                    ndcgs += self._calculate_ndcg(rank)
                    
                total_users += 1
                
        # Calculate final averages
        hr_at_10 = hits / total_users
        ndcg_at_10 = ndcgs / total_users
        
        print(f"\n{self.model.__class__.__name__} Results | HR@{top_k}: {hr_at_10:.4f} | NDCG@{top_k}: {ndcg_at_10:.4f}")
        return hr_at_10, ndcg_at_10

def evaluate_popularity_baseline(parquet_path, top_k=10): # Statistical baseline that ignores deep learning
    print("\nEvaluating Global Popularity Baseline...")
    
    # 1. Read data -> We need to find what the most globally popular tracks are
    df = pd.read_parquet(parquet_path, engine='pyarrow')
    
    # Flatten the training sequences to count how many times every song was played globally
    all_train_items = np.concatenate(df['train_sequence'].tolist())
    
    # Find the top 10 most popular songs in the entire training set
    print("Calculating global item frequencies...")
    popular_items = pd.Series(all_train_items).value_counts().head(top_k).index.values
    
    hits = 0.0
    total_users = len(df)
    
    # 2. Evaluation -> For every user, ignore their history and just recommend the global Top 10
    print("Scoring Popularity Baseline...")
    for test_item in df['test_item']:
        if test_item in popular_items:
            hits += 1.0 # The user actually listened to one of the global Top 10
            
    hr_at_10 = hits / total_users
    
    # Note: NDCG is technically computable here, but HR is the standard benchmark for the Pop baseline
    print(f"Popularity Baseline Results | HR@{top_k}: {hr_at_10:.4f}")
    return hr_at_10

def run():
    print("MRP Project: Sequential Recommender Evaluation Phase")
    
    # Initialize DataLoader -> We just need it to feed the test data and give us the dynamic vocab size
    loader = ds.SequentialDataLoader(file_path='./output/processed_sequences.parquet', max_len=50)
    dynamic_vocab_size = loader.total_items + 1 
    
    # --- 1. Evaluate SASRec ---
    sasrec_model = mdl.SASRec(vocab_size=dynamic_vocab_size, max_len=50, embed_dim=64, num_heads=2, num_blocks=2)
    
    # FIX: Explicitly build the graph in memory so Keras has "containers" to put the weights into
    # input_shape=(None, max_len) -> 'None' represents a dynamic batch size, '50' is our padded sequence length
    sasrec_model.build(input_shape=(None, 50)) 
    
    sasrec_model.load_weights('./saved_models/sasrec_final.weights.h5')
    
    sasrec_evaluator = RecommenderEvaluator(model=sasrec_model, dataset_loader=loader, total_items=loader.total_items)
    sasrec_hr, sasrec_ndcg = sasrec_evaluator.evaluate_model(top_k=10)
    
    # --- 2. Evaluate GRU4Rec ---
    gru_model = mdl.GRU4Rec(vocab_size=dynamic_vocab_size, max_len=50, embed_dim=64, gru_units=64)
    
    # FIX: Explicitly build the GRU graph in memory as well
    gru_model.build(input_shape=(None, 50))
    
    gru_model.load_weights('./saved_models/gru4rec_final.weights.h5')
    
    gru_evaluator = RecommenderEvaluator(model=gru_model, dataset_loader=loader, total_items=loader.total_items)
    gru_hr, gru_ndcg = gru_evaluator.evaluate_model(top_k=10)
    
    # --- 3. Evaluate Statistical Baseline ---
    pop_hr = evaluate_popularity_baseline('./output/processed_sequences.parquet', top_k=10)
    
    # --- 4. Final Output ---
    print("\n==========================================")
    print("FINAL MRP COMPARATIVE RESULTS (HR@10 & NDCG@10)")
    print("==========================================")
    print(f"1. SASRec (Transformer) : HR = {sasrec_hr:.4f} | NDCG = {sasrec_ndcg:.4f}")
    print(f"2. GRU4Rec (RNN)        : HR = {gru_hr:.4f} | NDCG = {gru_ndcg:.4f}")
    print(f"3. Popularity Baseline  : HR = {pop_hr:.4f} | NDCG = N/A")
    print("==========================================")

    pt.plot_evaluation_metrics(sasrec_hr, sasrec_ndcg, gru_hr, gru_ndcg, pop_hr)

if __name__ == "__main__":
    run()
