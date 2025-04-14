How to Run 'assignment_6.py'

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
