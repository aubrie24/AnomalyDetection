This project implements and compares multiple anomaly detection methods on the CelebA Bald vs. Non-Bald dataset, a highly imbalanced dataset. The goal is to detect rare cases (bald individuals) using both traditional and deep learning based approaches. 

The repository includes: 

Data Quality & Cleaning Scripts: Generate quality reports (missing values, duplicates, class imbalance, leakage checks). Removes duplicate rows and ensures numeric feature formatting. 

Feature Extraction Utilities: Functions to load datasets, separate features from labels, and save cleaned data. 

Model Training & Evaluation: Implements DevNet (semi supervised deep anomaly detection), Isolation Forest, KMeans, and a baseline random model. Evaluates models with AUC-ROC, Average Precision, and F1 Score. Includes score distribution plots and feature impportance ranking (Random Forest). 

Synthetic Data Testing: Demonstrates portability of saved models by predicting anomalies on artificial datasets. 

Key outputs included trained models (.pkl), evaluation reports, visualtions of anomaly score distributions, and top feauture rankings. 

How to Run 'anomaly_detection.py'

1. Create and activate a virtual environment (optional):

    python3 -m venv deepod-env
    source deepod-env/bin/activate

2. Install dependencies:
    pip install 'numpy<2.0' deepod pandas scikit-learn matplotlib joblib

3. Run full experiment using different models: 

    python main.py --model all
    python main.py --model devnet
    python main.py --model iforest
    python main.py --model cluster
    python main.py --model baseline

The following command will run all the models by default.
It is used to differentiate between running on the real data and running on the artificial data. 
However, it does not need to be explicitly defined as it is the default type:
    python main.py --type train_evaluation

5. Run artificial evaluation: 
    python main.py --type artificial

All outputs (plots, scores, saved models) are saved in the output/ directory.
