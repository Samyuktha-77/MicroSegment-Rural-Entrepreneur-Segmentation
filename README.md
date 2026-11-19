# MicroSegment – Indian Microfinance Loan Segmentation

## Overview
MicroSegment is an unsupervised machine learning project that explores patterns in Indian microfinance loan records and identifies groups with similar loan and business activity profiles.

The project uses clustering algorithms to support exploratory analysis of heterogeneous microfinance profiles.

## Objectives
- Analyse Indian microfinance loan records.
- Prepare a business-focused dataset.
- Perform feature engineering and preprocessing.
- Apply K-Means clustering.
- Evaluate clusters using internal validation metrics.
- Interpret cluster characteristics.
- Explore future advisory recommendations.

## Technology Stack
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Git and GitHub

## Dataset
Source: Kiva – Data Science for Good: Kiva Crowdfunding

https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding

The project uses 9,051 INR-denominated records in its business-focused analytical subset.

Note: These are historical Kiva loan records. They do not verify that every borrower is rural or that every loan is concessional credit.

## Methodology
1. Dataset inspection
2. India and INR filtering
3. Business-focused subset creation
4. Feature engineering
5. Data preprocessing
6. K-Means clustering
7. Cluster evaluation and interpretation

## Current K-Means Results

Selected candidate: K = 3

| Metric | Result |
|---|---:|
| Silhouette Score | 0.4924 |
| Davies-Bouldin Index | 0.8040 |
| Calinski-Harabasz Score | 8518.76 |

Cluster sizes:
- Cluster 0: 3,852 records
- Cluster 1: 3,966 records
- Cluster 2: 1,233 records

## Feature Ablation Experiment
After removing repayment interval features, K=3 achieved:
- Silhouette: 0.4427
- Davies-Bouldin: 0.9247
- Calinski-Harabasz: 6776.16

## Project Structure
- dataset/ – Local datasets (not included in Git)
- ml/ – Python scripts for preprocessing and clustering
- models/ – Locally saved trained models (not included in Git)
- results/ – Evaluation results and cluster summaries

## Limitations
- Historical data from 2014–2017.
- No verified rural-residence or concessional-credit indicator.
- No business revenue or profit information.
- Cluster interpretations require further validation.

## Future Work
- Agglomerative Clustering
- DBSCAN
- PCA visualisation
- Cluster stability analysis
- Transparent advisory module
- Web application and deployment

## Author
Samyuktha Padma

School of Computer Science and Engineering  
Lovely Professional University, Punjab, India
