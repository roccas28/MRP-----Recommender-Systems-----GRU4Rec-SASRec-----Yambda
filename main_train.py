# Roccas Raveendran - 500853856
# Sequential Recommender System - Machine Learning Orchestrator

import os
import time

# Custom Modules
import dataset as ds
import model as mdl
import train as tr
import plots as pt

def run():
    print("MRP Project: Sequential Recommender ML Training Phase")
    
    # Ensure checkpoint directory exists -> Prevents crash during final weight saving
    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    start_time = time.time()
    
    # --- PHASE 1: Data Ingestion ---
    print("\n--- Initializing Data Pipeline ---")
    # Hyperparameters: max_len=50 restricts context to last 50 interactions
    loader = ds.SequentialDataLoader(file_path='./output/processed_sequences.parquet', max_len=50)
    
    # Generate identical pipelines -> Shuffling enabled for training, disabled for testing
    train_dataset = loader.generate_tf_dataset(batch_size=256, is_training=True)
    test_dataset = loader.generate_tf_dataset(batch_size=256, is_training=False)
    
    # FIX: Dynamically extract the vocab_size from the loader
    # The +1 is mathematically mandatory to account for the '0' padding token!
    dynamic_vocab_size = loader.total_items + 1 
    
    # --- PHASE 2: Train SASRec (Transformer) ---
    print("\n--- Building & Training SASRec ---")
    # Instantiate the SASRec Transformer
    sasrec_model = mdl.SASRec(
        vocab_size=dynamic_vocab_size, 
        max_len=50, 
        embed_dim=64, 
        num_heads=2, 
        num_blocks=2, 
        dropout_rate=0.2
    )
    
    sasrec_trainer = tr.RecommenderTrainer(model=sasrec_model, train_dataset=train_dataset, val_dataset=test_dataset)
    
    # Hyperparameter: 10 epochs is standard for large-scale recommender systems before extreme overfitting occurs
    sasrec_history = sasrec_trainer.train(epochs=10)
    
    # Plot individual SASRec loss and save weights
    pt.plot_training_loss(sasrec_history, model_name="SASRec")
    sasrec_model.save_weights('./saved_models/sasrec_final.weights.h5')
    
    # --- PHASE 3: Train GRU4Rec (Recurrent Neural Network) ---
    print("\n--- Building & Training GRU4Rec ---")
    # Instantiate the GRU model -> Uses same embed_dim as SASRec to ensure a mathematically fair comparison
    gru_model = mdl.GRU4Rec(
        vocab_size=dynamic_vocab_size, 
        max_len=50, 
        embed_dim=64, 
        gru_units=64, 
        dropout_rate=0.2
    )
    
    gru_trainer = tr.RecommenderTrainer(model=gru_model, train_dataset=train_dataset, val_dataset=test_dataset)
    
    # Train for the exact same number of epochs
    gru_history = gru_trainer.train(epochs=10)
    
    # Plot individual GRU4Rec loss and save weights
    pt.plot_training_loss(gru_history, model_name="GRU4Rec")
    gru_model.save_weights('./saved_models/gru4rec_final.weights.h5')
    
    # --- PHASE 4: Comparative Plotting ---
    print("\n--- Generating Comparative Visualizations ---")
    # Execute new comparative plot to show which architecture converged better
    pt.plot_comparative_loss(sasrec_history, gru_history)
    
    print(f"\nML Pipeline Complete! Total Time elapsed: {(time.time() - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    run() # Execute main block
