"""
Main script for FOMC topic modeling and influence analysis.
"""

import pandas as pd
from . import (
    TopicModeler,
    InfluenceAnalyzer,
    PolicyAnalyzer,
    plot_topic_distribution,
    plot_influence_heatmap,
    plot_topic_description_heatmap
)

def main():
    # Initialize models
    topic_modeler = TopicModeler(num_topics=40)
    influence_analyzer = InfluenceAnalyzer()
    policy_analyzer = PolicyAnalyzer()
    
    # Load and process documents
    docs = pd.read_csv("speaker_speech.csv")
    
    # Clean the data
    print(f"Original document count: {len(docs)}")
    docs = docs.dropna(subset=['Speech'])  # Remove rows with NaN in Speech column
    print(f"Document count after removing NaN: {len(docs)}")
    
    # Convert Speech column to string type
    docs['Speech'] = docs['Speech'].astype(str)
    
    corpus = topic_modeler.prepare_corpus(docs['Speech'])
    
    # Train topic model
    topic_modeler.train(corpus)
    
    # First perform policy analysis to identify relevant topics
    policy_analyzer.load_data("cleaned_agg_topic_output.csv", "controls.csv")
    policy_analyzer.lasso_analysis_for_section("fomc1")
    
    # Get significant policy topics (those selected in >50% of runs)
    significant_topics = policy_analyzer.get_significant_topics(threshold=50)
    policy_topics = list(significant_topics.keys())
    
    # Then analyze influence using only the policy-relevant topics
    influence_analyzer.load_data("cleaned_agg_topic_output.csv", "controls.csv")
    
    # Filter problematic meetings
    drop_dates = [19880329, 19880517, 19910514, 20040630, 20080916]
    influence_analyzer.filter_meetings(drop_dates)
    
    # Analyze all meetings using only policy topics
    influence_analyzer.analyze_all_meetings(policy_topics)
    
    # Save results
    topic_modeler.save_model("models/lda_model")
    
    # Create visualizations
    # Plot topic distributions for a sample document
    sample_doc = docs['Speech'].iloc[0]
    topic_dist = topic_modeler.get_document_topics(sample_doc)
    topic_names = [f"Topic {i}" for i in range(topic_modeler.num_topics)]
    plot_topic_distribution(topic_dist, topic_names, title="Sample Document Topic Distribution")
    
    # Plot influence heatmap for a sample meeting
    sample_date = influence_analyzer.df['Date'].iloc[0]
    influence_matrix = influence_analyzer.get_influence_matrix(sample_date)
    if influence_matrix is not None:
        speaker_names = influence_analyzer.df[influence_analyzer.df['Date'] == sample_date]['Speaker'].unique()
        plot_influence_heatmap(influence_matrix, speaker_names, title=f"Influence Matrix for Meeting {sample_date}")
    
    # Plot topic description heatmap
    plot_topic_description_heatmap("topic_description_sorted.csv", policy_topics=range(15))

if __name__ == "__main__":
    main() 