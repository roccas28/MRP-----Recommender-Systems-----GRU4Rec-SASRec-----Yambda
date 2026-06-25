# Roccas Raveendran - 500853856
# Sequential Recommender System - Custom Training Loop & Loss Functions

import tensorflow as tf
import numpy as np

class RecommenderTrainer: # Orchestrates custom gradient descent 
    
    def __init__(self, model, train_dataset, val_dataset, optimizer=None):
        print("Initializing Custom Training Loop...")
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Hyperparameter: Learning Rate -> 0.001 is the industry standard starting point for Adam
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    @tf.function # Compiles the python function into a static C++ graph -> Massively accelerates batch training
    def _train_step(self, inputs, targets):
        
        with tf.GradientTape() as tape:
            # 1. Forward Pass -> Extract the user's current mood vector from their padded history
            user_state = self.model(inputs['input_sequence'], training=True) 
            
            # 2. Extract Target Embeddings -> Retrieve the dense mathematical vectors for the items
            pos_emb = self.model.item_embedding(targets['positive_target'])
            neg_emb = self.model.item_embedding(targets['negative_target'])
            
            # 3. Calculate Dot Products -> Measures cosine-like similarity between user mood and candidate items
            pos_scores = tf.reduce_sum(user_state * pos_emb, axis=-1)
            neg_scores = tf.reduce_sum(user_state * neg_emb, axis=-1)
            
            # 4. Bayesian Personalized Ranking (BPR) Loss -> Forces the true item score to be higher than the noise
            # Note: 1e-9 is added to prevent math domain errors if log encounters a perfect 0
            loss = -tf.reduce_mean(tf.math.log(tf.sigmoid(pos_scores - neg_scores) + 1e-9))
            
        # 5. Backpropagation -> Calculate gradients based on BPR error and update network weights
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss

    def train(self, epochs=10): # Main execution loop for the training phase
        print(f"Beginning Training for {epochs} Epochs...")
        
        history = {'loss': []} # Track metrics for plotting -> Stores average loss per epoch
        
        for epoch in range(epochs):
            total_loss = 0.0
            steps = 0
            
            # Iterate through the tf.data.Dataset generator stream
            for inputs, targets in self.train_dataset:
                loss = self._train_step(inputs, targets) # Execute compiled gradient step
                total_loss += float(loss) # Extract scalar from tensor -> Accumulates total error
                steps += 1
                
            avg_loss = total_loss / steps # Calculate mean error across all batches
            history['loss'].append(avg_loss)
            
            print(f"Epoch {epoch + 1}/{epochs} | BPR Loss: {avg_loss:.4f}")
            
        return history
