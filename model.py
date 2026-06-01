# Roccas Raveendran - 500853856
# Sequential Recommender System - Core Neural Architectures (SASRec & GRU4Rec)

import tensorflow as tf
from tensorflow.keras import layers, Model

# ==========================================
# TRANSFORMER COMPONENTS (SASRec)
# ==========================================

class TransformerBlock(layers.Layer): # Custom Modular Layer -> Builds the core Self-Attention mechanism
    def __init__(self, embed_dim, num_heads, dropout_rate=0.2, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        
        # 1. Multi-Head Attention Layer
        # Hyperparameter: num_heads -> Allows the model to focus on different temporal patterns simultaneously
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate)
        
        # 2. Feed-Forward Network (FFN) -> Non-linear projection after attention
        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim, activation="relu"), # Projects to same dimensionality
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim)
        ])
        
        # 3. Layer Normalization -> Stabilizes the gradient descent during deep network training
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # Apply Self-Attention with Causal Masking -> 'use_causal_mask=True' mathematically blinds the model to the future
        attn_output = self.attention(inputs, inputs, use_causal_mask=True, training=training)
        
        # First Residual Connection + Norm -> Prevents vanishing gradients
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-Forward + Second Residual Connection + Norm
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class SASRec(Model): # Core Transformer Recommender Architecture
    def __init__(self, vocab_size, max_len=50, embed_dim=64, num_heads=2, num_blocks=2, dropout_rate=0.2, **kwargs):
        super(SASRec, self).__init__(**kwargs)
        
        print(f"Initializing SASRec: embed_dim={embed_dim}, num_heads={num_heads}, num_blocks={num_blocks}")
        self.max_len = max_len
        self.vocab_size = vocab_size # Total unique items + 1 (for padding token 0)
        
        # Item Embedding Layer -> Maps sparse integer IDs to dense continuous vectors
        self.item_embedding = layers.Embedding(
            input_dim=vocab_size, 
            output_dim=embed_dim, 
            mask_zero=True # Instructs TF to mathematically ignore our '0' padding tokens
        )
        
        # Positional Embedding Layer -> Injects the concept of "time" into the Transformer
        self.pos_embedding = layers.Embedding(input_dim=max_len, output_dim=embed_dim)
        
        # Dropout to prevent overfitting on specific user sequences
        self.emb_dropout = layers.Dropout(dropout_rate)
        
        # Stack multiple Transformer Blocks -> Hyperparameter: num_blocks controls depth of pattern recognition
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, dropout_rate) for _ in range(num_blocks)
        ]

    def call(self, inputs, training=False):
        # inputs shape: (batch_size, max_len)
        
        # 1. Get Item Embeddings
        seq_embeddings = self.item_embedding(inputs) # Shape: (batch_size, max_len, embed_dim)
        
        # 2. Get Positional Embeddings -> Create an array of positions [0, 1, 2, ..., max_len-1]
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        pos_embeddings = self.pos_embedding(positions) # Shape: (max_len, embed_dim)
        
        # 3. Combine Item and Positional Embeddings -> The network now knows WHAT the item is and WHEN it was played
        x = seq_embeddings + pos_embeddings
        x = self.emb_dropout(x, training=training)
        
        # 4. Pass through the stacked Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)
            
        # 5. Extract the Final Hidden State -> We slice the array to grab the absolute LAST item in the sequence
        # Because we padded on the left, x[:, -1, :] represents the user's entire aggregated history right before the prediction
        final_user_state = x[:, -1, :] # Shape: (batch_size, embed_dim)
        
        return final_user_state

# ==========================================
# RECURRENT COMPONENTS (GRU4Rec)
# ==========================================

class GRU4Rec(Model): # Baseline Recurrent Neural Network Architecture
    def __init__(self, vocab_size, max_len=50, embed_dim=64, gru_units=64, dropout_rate=0.2, **kwargs):
        super(GRU4Rec, self).__init__(**kwargs)
        
        print(f"Initializing GRU4Rec: embed_dim={embed_dim}, gru_units={gru_units}")
        self.vocab_size = vocab_size
        
        # Item Embedding Layer
        self.item_embedding = layers.Embedding(
            input_dim=vocab_size, 
            output_dim=embed_dim, 
            mask_zero=True 
        )
        
        # GRU Layer -> Stateful recurrent layer that naturally processes sequences left-to-right
        self.gru = layers.GRU(
            units=gru_units, 
            return_sequences=False, # We only want the final aggregated state, not the state at every time-step
            dropout=dropout_rate
        )

    def call(self, inputs, training=False):
        # inputs shape: (batch_size, max_len)
        
        # 1. Get Item Embeddings -> Notice there are NO positional embeddings required for RNNs
        x = self.item_embedding(inputs)
        
        # 2. Pass through GRU -> The recurrent nature inherently understands time/order
        final_user_state = self.gru(x, training=training) # Shape: (batch_size, gru_units)
        
        return final_user_state
