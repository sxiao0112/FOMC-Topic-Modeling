"""
Policy topic analysis for FOMC meetings.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold

class PolicyAnalyzer:
    def __init__(self, n_runs=100):
        self.n_runs = n_runs
        self.selection_counts = {}
        self.coefficients = {}
        
    def load_data(self, topic_output_path, shift_data_path):
        """Load and prepare data for policy analysis."""
        data_controls = pd.read_csv(shift_data_path)
        data_csv = pd.read_csv(topic_output_path)
        
        # Rename columns and merge datasets
        data_controls.rename(columns={"Time": "Date", "Name": "Speaker"}, inplace=True)
        data_csv.rename(columns={"date": "Date", "speaker": "Speaker", "section": "Section"}, inplace=True)
        self.merged_data = pd.merge(data_csv, data_controls, on=["Date", "Speaker"], how="left")
    
    def lasso_analysis_for_section(self, section, topics=None, controls=None):
        """Perform LASSO regression analysis for a specific meeting section."""
        if topics is None:
            topics = [f"T{i}" for i in range(15)]
        if controls is None:
            controls = ["CPI_Nation_Con", "Unemployment_Nation_Con"]
            
        # Filter data
        filtered_data = self.merged_data[
            self.merged_data["Section"] == section
        ].dropna(subset=["Dissent", "Voting_Status"] + controls)
        
        # Extract features
        X = filtered_data[topics + controls]
        y = filtered_data["Dissent"]
        
        # Initialize tracking
        self.selection_counts = {topic: 0 for topic in topics}
        coef_sums = {topic: 0 for topic in topics}
        
        # Perform multiple runs
        for _ in range(self.n_runs):
            # Define and fit model
            model = ElasticNetCV(
                l1_ratio=1,  # Lasso regression
                cv=KFold(n_splits=5, shuffle=True),
                random_state=np.random.randint(0, 1000)
            )
            model.fit(X, y)
            
            # Track selections and coefficients
            for topic, coef in zip(topics, model.coef_[:len(topics)]):
                if abs(coef) > 1e-5:  # Non-zero coefficient
                    self.selection_counts[topic] += 1
                    coef_sums[topic] += coef
        
        # Calculate average coefficients for selected topics
        self.coefficients = {
            topic: coef_sums[topic] / max(count, 1)
            for topic, count in self.selection_counts.items()
        }
        
        return self.selection_counts, self.coefficients
    
    def get_significant_topics(self, threshold=50):
        """Get topics that were selected more than threshold times."""
        return {
            topic: count
            for topic, count in self.selection_counts.items()
            if count >= threshold
        }
    
    def get_topic_impact(self, topic):
        """Get the impact (coefficient) of a specific topic."""
        return self.coefficients.get(topic, 0) 