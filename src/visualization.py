"""
Visualization utilities for topic modeling and influence analysis.
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import pandas as pd

def plot_topic_distribution(topic_dist, topic_names, title="Topic Distribution"):
    """Plot topic distribution using matplotlib."""
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(topic_dist)), topic_dist)
    plt.xticks(range(len(topic_dist)), topic_names, rotation=45)
    plt.title(title)
    plt.xlabel("Topics")
    plt.ylabel("Probability")
    plt.tight_layout()
    return plt.gcf()

def plot_influence_heatmap(influence_matrix, speaker_names, title="Influence Matrix"):
    """Plot influence matrix as a heatmap using plotly."""
    fig = go.Figure(data=go.Heatmap(
        z=influence_matrix,
        x=speaker_names,
        y=speaker_names,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Speaker",
        yaxis_title="Speaker",
        width=800,
        height=800
    )
    return fig

def plot_difference_matrix(mdiff, title="", annotation=None):
    """Plot difference matrix with optional annotations."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mdiff, cmap='RdBu', aspect='auto')
    plt.colorbar(im)
    
    if annotation:
        ax.set_xticks(range(len(annotation)))
        ax.set_yticks(range(len(annotation)))
        ax.set_xticklabels(annotation, rotation=90)
        ax.set_yticklabels(annotation)
    
    ax.set_title(title)
    plt.tight_layout()
    return fig

def plot_time_series(dates, values, title="", ylabel="Value"):
    """Plot time series data."""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, values, marker='o')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()

def plot_topic_description_heatmap(topic_description_path, policy_topics=None, figsize=(65, 27)):
    """Plot topic description heatmap."""
    # Load and prepare data
    df = pd.read_csv(topic_description_path, header=1)
    df = df.dropna(axis=1, how='all')
    df = df.iloc[:, :11]
    df = df.T
    
    # Set column names
    df.columns = [f"topic{int(i/2)}" if i % 2 == 0 else "" for i in range(df.shape[1])]
    df = df.iloc[1:]
    
    # Split into topics and probabilities
    topics_df = df.iloc[:, ::2].reset_index(drop=True)
    probs_df = df.iloc[:, 1::2].reset_index(drop=True)
    
    # Filter for policy topics if specified
    if policy_topics is not None:
        topics_df = topics_df.iloc[:, policy_topics]
        probs_df = probs_df.iloc[:, policy_topics]
    
    # Convert probabilities to numeric
    probs_df = probs_df.apply(pd.to_numeric)
    
    # Transpose for visualization
    topics_df = topics_df.T
    probs_df = probs_df.T
    
    # Create labels
    y_axis_labels = [f"topic {i}" for i in (policy_topics if policy_topics else range(topics_df.shape[0]))]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        probs_df,
        cmap='Blues',
        ax=ax,
        annot=topics_df,
        annot_kws={"size": 30},
        fmt='',
        vmax=0.2,
        xticklabels=False,
        yticklabels=y_axis_labels
    )
    
    # Customize appearance
    ax.tick_params(left=False, bottom=False)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig 