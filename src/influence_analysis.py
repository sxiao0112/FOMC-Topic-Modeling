"""
Analysis of influence between committee members in FOMC meetings.

Following Hansen et al. (2018), the influence of a Reserve Bank president on the entire committee 
is measured as the similarity between the policy topic coverage of the individual’s speech during 
the economic outlook discussion (FOMC1) and the policy topic coverage of all committee members’ speeches 
during the monetary policy decision discussion (FOMC2) of the same meeting.

Three established approaches from the literature are usedas proxies for the similarity:
1. Bhattacharyya coefficient
2. Dot product
3. Kullback-Leibler divergence
"""

import pandas as pd
import numpy as np
from . import utils

class InfluenceAnalyzer:
    def __init__(self):
        self.influence_matrices = {
            'bhattacharyya': {},
            'dot_product': {},
            'kl_divergence': {}
        }
    
    def load_data(self, topic_output_path, shift_data_path=None):
        """Load and prepare data for influence analysis."""
        self.df = pd.read_csv(topic_output_path)
        if shift_data_path:
            self.shift_df = pd.read_csv(shift_data_path)
    
    def filter_meetings(self, drop_dates):
        """Filter out specific meeting dates."""
        self.df = self.df[~self.df['Date'].isin(drop_dates)]
    
    def compute_bhattacharyya_coefficient(self, p, q):
        """Compute Bhattacharyya coefficient between two distributions."""
        return np.sum(np.sqrt(p * q))
    
    def compute_kl_divergence(self, p, q):
        """Compute KL divergence between two distributions."""
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        return np.sum(p * np.log(p / q))
    
    def compute_influence_matrices(self, topics, date_group):
        """
        Compute influence matrices for a specific meeting using only policy-relevant topics.
        
        Args:
            topics (list): List of policy-relevant topic columns (e.g., ['T1', 'T3', 'T5'])
            date_group (DataFrame): Data for a specific meeting date
        """
        fomc1 = date_group[date_group['Section'] == 'fomc1']
        fomc2 = date_group[date_group['Section'] == 'fomc2']
        
        # Get shared speakers
        shared_speakers = list(set(fomc1['Speaker']) & set(fomc2['Speaker']))
        n_speakers = len(shared_speakers)
        
        # Initialize matrices
        B_matrix = np.zeros((n_speakers, n_speakers))
        D_matrix = np.zeros((n_speakers, n_speakers))
        KL_matrix = np.zeros((n_speakers, n_speakers))
        
        for i, speaker_i in enumerate(shared_speakers):
            # Extract only policy-relevant topic distributions
            p1 = fomc1[fomc1['Speaker'] == speaker_i][topics].values[0]
            p2 = fomc2[fomc2['Speaker'] == speaker_i][topics].values[0]
            
            # Normalize the distributions to sum to 1 (since we're using a subset of topics)
            p1 = p1 / p1.sum()
            p2 = p2 / p2.sum()
            
            for j, speaker_j in enumerate(shared_speakers):
                q1 = fomc1[fomc1['Speaker'] == speaker_j][topics].values[0]
                q1 = q1 / q1.sum()  # Normalize
                
                # Compute similarities using only policy-relevant topics
                B_matrix[i, j] = self.compute_bhattacharyya_coefficient(p2, q1)
                D_matrix[i, j] = np.dot(p2, q1)
                KL_matrix[i, j] = self.compute_kl_divergence(p2, q1)
        
        return {
            'bhattacharyya': utils.normalize_matrix_column(B_matrix),
            'dot_product': utils.normalize_matrix_column(D_matrix),
            'kl_divergence': utils.normalize_matrix_column(KL_matrix)
        }
    
    def analyze_all_meetings(self, topics):
        """Analyze influence patterns for all meetings."""
        for date, group in self.df.groupby('Date'):
            self.influence_matrices['bhattacharyya'][date] = self.compute_influence_matrices(topics, group)
    
    def get_influence_matrix(self, date, method='bhattacharyya'):
        """Get influence matrix for a specific date and method."""
        return self.influence_matrices[method].get(date) 