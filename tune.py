# Roccas Raveendran - 500853856
# Sequential Recommender System - Comprehensive 3D Grid Search

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import time
import os

# Custom Modules
import dataset as ds
import model as mdl
import train as tr
import evaluate as ev
import plots as pt

def run_grid_search():
    print("Initiating Comprehensive 3D Grid Search (SASRec vs GRU4Rec)...")
    
    # 1. Expanded Hyperparameter Grid
    max_len_options = [10, 30, 50, 70, 100] 
    embed_dim_options = [16, 32, 64, 128, 256]
    learning_rates = [0.0005, 0.001, 0.005]
    models_to_test = ['SASRec', 'GRU4Rec']
    
    results = [] 
    
    dummy_loader = ds.SequentialDataLoader(file_path='./output/processed_sequences.parquet', max_len=50)
    dynamic_vocab_size = dummy_loader.total_items + 1
    
    total_runs = len(max_len_options) * len(embed_dim_options) * len(learning_rates) * len(models_to_test)
    current_run = 1
    
    for seq_len in max_len_options:
        # Re-instantiate dataloader for new sequence length padding
        loader = ds.SequentialDataLoader(file_path='./output/processed_sequences.parquet', max_len=seq_len)
        train_dataset = loader.generate_tf_dataset(batch_size=256, is_training=True)
        
        for dim in embed_dim_options:
            for lr in learning_rates:
                for arch in models_to_test:
                    print(f"\n==========================================================")
                    print(f"Run {current_run}/{total_runs} | Model: {arch} | K: {seq_len} | Dim: {dim} | LR: {lr}")
                    print(f"==========================================================")
                    
                    # Instantiate chosen architecture
                    if arch == 'SASRec':
                        model = mdl.SASRec(vocab_size=dynamic_vocab_size, max_len=seq_len, embed_dim=dim, num_heads=2, num_blocks=2, dropout_rate=0.2)
                    else:
                        model = mdl.GRU4Rec(vocab_size=dynamic_vocab_size, max_len=seq_len, embed_dim=dim, gru_units=dim, dropout_rate=0.2)
                    
                    # Train model with specific learning rate
                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                    trainer = tr.RecommenderTrainer(model=model, train_dataset=train_dataset, val_dataset=None, optimizer=optimizer)
                    trainer.train(epochs=10) 
                    
                    # Evaluate model
                    model.build(input_shape=(None, seq_len))
                    evaluator = ev.RecommenderEvaluator(model=model, dataset_loader=loader, total_items=loader.total_items)
                    hr, ndcg = evaluator.evaluate_model(top_k=10)
                    
                    # Log comprehensive results
                    results.append({
                        'Model': arch,
                        'Sequence_Length': seq_len,
                        'Embedding_Dim': dim,
                        'Learning_Rate': lr,
                        'HR@10': float(hr),
                        'NDCG@10': float(ndcg)
                    })
                    
                    current_run += 1
                    
                    # VITAL: Clear VRAM to prevent memory overflow during the 150-run loop
                    tf.keras.backend.clear_session()
            
    # Export Results
    print("\n--- Grid Search Complete. Generating Artifacts ---")
    results_df = pd.DataFrame(results)
    results_df.to_csv('./output/comprehensive_grid_search.csv', index=False)
    
    # Generate Heatmaps & Line Graphs for both models
    pt.plot_dual_hyperparameter_heatmaps(results_df, metric='HR@10')

    pt.plot_hyperparameter_line_graphs(results_df, metric='HR@10')

if __name__ == "__main__":
    run_grid_search()
