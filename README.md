# FOMC Topic Modeling and Influence Analysis

This project analyzes Federal Open Market Committee (FOMC) meeting transcripts using topic modeling and measures the influence between speakers. Following Hansen et al. (2018), influence is calculated as the similarity between the topic distribution of each committee member's speeches and those of the rest of the committee at each FOMC meeting.

Data is not included in this repository.

## Project Structure

```
.
├── input/                   # Input data directory
│   ├── speaker_speech.csv   # Meeting speeches data
│   └── controls.csv         # Speaker backgrounds & economic indicators
├── output/                  # Output data directory
│   ├── topic_description.csv
│   ├── agg_topic_output.csv
│   ├── within_meeting_influence_full.csv
│   └── models/             # Trained models directory
│       └── lda_model       # Saved topic model
├── src/
│   ├── __init__.py
│   ├── main.py               # Main entry point
│   ├── topic_modeling.py     # Topic modeling training
│   ├── influence_analysis.py # Influence pattern analysis
│   ├── policy_analysis.py    # Policy topic analysis
│   ├── visualization.py      # Visualization utilities
│   └── utils.py              # Common utilities
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Input Files (place in input/ directory)

1. `speaker_speech.csv`: Contains FOMC meeting speeches with columns:
   - Date: Meeting date (YYYYMMDD format)
   - Speaker: Speaker's name
   - Speech: Speech text content (with appendix)
   - Section: Meeting section ("fomc1" economics outlook/"fomc2" policy go-round)

2. `controls.csv`: Contains information about speaker backgrounds and economic indicators:
   - Time: Meeting date
   - Name: Speaker name
   - Status: Speaker status
   - Additional economic indicators (CPI, Unemployment, etc.)

## Output Files (generated in output/ directory)

1. `topic_description.csv`: Describes the selected topics and their composition
   - Contains topic words and their probabilities
   - Used for topic interpretation and visualization

2. `agg_topic_output.csv`: Topic distribution for each speaker at each meeting section
   - Contains speaker-wise topic distributions
   - Used for influence analysis

3. `within_meeting_influence_full.csv`: Influence patterns between speakers
   - Generated from influence analysis
   - Contains influence scores between speakers

4. `models/lda_model`: Saved trained topic model
   - Can be loaded for future analysis
   - Contains model parameters and vocabulary

## Important Notes

1. Meeting Exclusions:
   - The following meetings are dropped due to lack of key information:
     * 19880329
     * 19880517
     * 19910514
     * 20040630
     * 20080916

2. Policy Analysis:
   - Uses LASSO regression to identify significant topics
   - Controls for economic indicators (CPI, Unemployment)
   - Performs multiple runs for robust feature selection

3. Influence Analysis:
   - Only considers shared speakers present in both fomc1 and fomc2 sections
   - The sum of influence scores within each meeting is normalized to 1
   - Self-influence is included in the analysis

## Module Description

- `topic_modeling.py`: Implements LDA topic modeling
  * Handles document preprocessing
  * Trains topic models
  * Extracts topic distributions

- `influence_analysis.py`: Analyzes speaker influence 
  * Computes influence measures using multiple similarity measures
  * Handles temporal analysis

- `policy_analysis.py`: Analyzes policy-related topics
  * Performs LASSO regression analysis
  * Quantifies topic impact on decisions
  * Identifies significant topics related to monetary policy

- `visualization.py`: Provides plotting utilities
  * Topic distribution plots
  * Influence heatmaps

- `utils.py`: Common utilities
  * Text preprocessing
  * Data loading
  * Matrix operations

## Usage

Run the main analysis:
```bash
python -m src.main
```

## Dependencies

- numpy: Numerical computations
- pandas: Data manipulation
- matplotlib: Basic plotting
- nltk: Text processing
- gensim: Topic modeling
- plotly: Interactive visualizations
- scikit-learn: Machine learning utilities
- seaborn: Statistical visualizations 

## References

Hansen, S., McMahon, M., & Prat, A. (2018). Transparency and deliberation within the FOMC: A computational linguistics approach. Quarterly Journal of Economics, 133(2), 801-870.
