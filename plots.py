# Roccas Raveendran - 500853856
# Sequential Recommender System - Exploratory Data Analysis (EDA) Plots

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set global aesthetics -> Matches formatting from previous clinical/deep learning projects
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12, 
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

def plot_played_ratio_distribution(df): # Plots implicit feedback completion rates
    print("Generating played ratio histogram...")
    plt.figure(figsize=(10, 6)) # Initialize figure size -> Standard widescreen aspect
    
    # Plot histogram -> Hyperparameter: bins=50 provides enough granularity to see the bimodal peaks
    sns.histplot(data=df, x="played_ratio_pct", bins=50, color="b", kde=False)
    
    # Draw reference lines for feedback thresholds -> Visually justifies the Hard Filter bounds
    plt.axvline(x=10, color='r', linestyle='--', label="Skip Threshold (<10%)")
    plt.axvline(x=70, color='g', linestyle='--', label="Listen Threshold (>70%)")
    
    plt.title("Distribution of Track Completion Rates (Implicit Feedback)")
    plt.ylabel("Frequency")
    plt.xlabel("Played Ratio (%)")
    plt.legend(loc='upper center')
    plt.tight_layout() # Compress margins -> Prevents label cutoff
    plt.savefig("./Output/eda_played_ratio.png", dpi=300) # Export high-res PNG
    print("Saved eda_played_ratio.png")

def plot_long_tail_popularity(df): # Plots item popularity to justify negative sampling
    print("Generating long-tail popularity curve...")
    
    item_counts = df['item_id'].value_counts().values # Count frequency of each item -> Arrays automatically sorted descending
    
    plt.figure(figsize=(10, 6)) # Initialize figure size
    plt.plot(item_counts, color="purple", linewidth=2) # Line plot showcases the steep drop-off
    
    plt.title("Long-Tail Item Popularity Distribution")
    plt.ylabel("Number of Interactions")
    plt.xlabel("Items (Sorted by Popularity Rank)")
    plt.yscale('log') # Hyperparameter/Design: Log scale is mandatory -> Condenses 900k items into readable variance
    plt.tight_layout()
    plt.savefig("./Output/eda_long_tail.png", dpi=300)
    print("Saved eda_long_tail.png")

def plot_sequence_length_distribution(df_lengths): # Plots distribution of user history lengths
    print("Generating sequence length distribution...")
    plt.figure(figsize=(10, 6))
    
    # Plot histogram -> Hyperparameter: bins=100 provides fine detail on the extreme right-skewed outliers
    sns.histplot(data=df_lengths, x="seq_length", bins=100, color="teal", kde=False)
    
    plt.title("Distribution of User History Lengths")
    plt.ylabel("Number of Users (Log Scale)")
    plt.xlabel("Sequence Length (Total Interactions)")
    plt.yscale('log') # Log scale -> Highlights extreme power users (ex: users with 10k+ interactions)
    plt.tight_layout()
    plt.savefig("./output/eda_sequence_lengths.png", dpi=300)
    print("Saved eda_sequence_lengths.png")

def plot_event_type_distribution(df_events): # Plots interaction composition breakdown
    print("Generating event type distribution...")
    plt.figure(figsize=(10, 6))
    
    df_events = df_events.sort_values('total_count', ascending=False) # Sort descending -> Creates clean waterfall aesthetic
    
    # Plot bar chart -> Visualizes ratio of passive vs active events
    sns.barplot(data=df_events, x="event_type", y="total_count", palette="viridis") 
    
    plt.title("Total Interactions by Event Type")
    plt.ylabel("Total Count (Tens of Millions)")
    plt.xlabel("Event Type")
    plt.tight_layout()
    plt.savefig("./output/eda_event_types.png", dpi=300)
    print("Saved eda_event_types.png")
