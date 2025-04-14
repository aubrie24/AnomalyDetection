import utils
from sklearn.model_selection import train_test_split
from deepod.models.tabular import DevNet
from deepod.metrics import tabular_metrics
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
import importlib
import sys
import time

#Authors: Aubrie Pressley & Lisette Kamper-Hinson
#Acknowledgements: Chat GPT

#function to make sure user has the correct downloads
#written by chatgpt
def check_dependencies():
    try:
        import numpy
        from packaging import version
        if version.parse(numpy.__version__) >= version.parse("2.0"):
            print(f"[ERROR] NumPy version {numpy.__version__} is incompatible with DeepOD.")
            print("Please downgrade NumPy to a version < 2.0 using:")
            print("    pip install 'numpy<2.0'")
            sys.exit(1)
    except ImportError:
        print("[ERROR] NumPy is not installed.")
        print("Please install it using: pip install 'numpy<2.0'")
        sys.exit(1)

    if importlib.util.find_spec("deepod") is None:
        print("[ERROR] The 'deepod' package is not installed.")
        print("Please install it using: pip install deepod")
        sys.exit(1)

def feature_importance(features, labels):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(features, labels)

    importances = clf.feature_importances_
    feature_names = features.columns

    #data frame with feature and importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("Feature importances saved to '../output/top_features_rf.csv'")
    importance_df.to_csv("../output/top_features_rf.csv", index=False)

def plot_score_distribution(scores, labels, model_name):
    plt.figure(figsize=(8, 5))
    plt.hist(scores[labels == 0], bins=50, alpha=0.6, label='Non-Bald (0)')
    plt.hist(scores[labels == 1], bins=50, alpha=0.6, label='Bald (1)', color='blue')
    plt.title(f"Anomaly Score Distribution - {model_name}")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../output/{model_name.lower()}_score_distribution.png")
    plt.show()

def baseline(train_features, train_labels, dev_features=None, dev_labels=None, retrain=False):
    print(f'using dummy for anomaly detection (random method).')
    output_file = "../output/baseline_results.txt"

    np.random.seed(42)
    start_pred = time.time()
    train_scores = np.random.rand(len(train_labels))
    dev_scores = np.random.rand(len(dev_labels))

    # threshold to binarize scores into predictions (top 2.2% as anomalies)
    train_preds = train_scores >= np.percentile(train_scores, 100 - 2.2)
    dev_preds = dev_scores >= np.percentile(dev_scores, 100 - 2.2)

    pred_time = time.time() - start_pred

    # metrics
    train_auc = roc_auc_score(train_labels, train_scores)
    train_ap = average_precision_score(train_labels, train_scores)
    train_f1 = f1_score(train_labels, train_preds)

    dev_auc = roc_auc_score(dev_labels, dev_scores)
    dev_ap = average_precision_score(dev_labels, dev_scores)
    dev_f1 = f1_score(dev_labels, dev_preds)

    with open(output_file, 'w') as f:
        f.write("[Train Set Evaluation - Dummy (random assignment)]\n")
        f.write(f"AUC: {train_auc:.4f}\n")
        f.write(f"Average Precision: {train_ap:.4f}\n")
        f.write(f"F1 Score: {train_f1:.4f}\n\n")
        f.write("[Dev Set Evaluation - Dummy (random assignment)]\n")
        f.write(f"AUC: {dev_auc:.4f}\n")
        f.write(f"Average Precision: {dev_ap:.4f}\n")
        f.write(f"F1 Score: {dev_f1:.4f}\n")
        f.write(f"\nPrediction time: {pred_time}")

    # plot distribution for dev set
    plot_score_distribution(dev_scores, dev_labels, "Baseline")

    return ('Baseline', dev_auc, dev_ap, None, None)

def iforest(train_features, train_labels, dev_features=None, dev_labels=None, retrain=False):
    print(f'using iforest for anomaly detection')
    output_file = "../output/iforest_results.txt"

    clf = IsolationForest(contamination=0.022, random_state=42)
    train_start = time.time()
    clf.fit(train_features)
    train_time = time.time() - train_start

    if retrain:
        return ('Isolation Forest', None, None, clf, None)

    #train set eval
    train_scores = -clf.decision_function(train_features)
    train_scores = MinMaxScaler().fit_transform(train_scores.reshape(-1, 1)).flatten()
    train_auc = roc_auc_score(train_labels, train_scores)
    train_ap = average_precision_score(train_labels, train_scores)
    train_preds = train_scores >= np.percentile(train_scores, 100 - 2.2)
    train_f1 = f1_score(train_labels, train_preds)

    #dev set eval
    pred_start = time.time()
    dev_scores = -clf.decision_function(dev_features)
    dev_scores = MinMaxScaler().fit_transform(dev_scores.reshape(-1, 1)).flatten()
    pred_time = time.time() - pred_start
    dev_auc = roc_auc_score(dev_labels, dev_scores)
    dev_ap = average_precision_score(dev_labels, dev_scores)
    dev_preds = dev_scores >= np.percentile(dev_scores, 100 - 2.2)
    dev_f1 = f1_score(dev_labels, dev_preds)

    with open(output_file, 'w') as f:
        f.write("[Train Set Evaluation - Isolation Forest]\n")
        f.write(f"AUC: {train_auc:.4f}\n")
        f.write(f"Average Precision: {train_ap:.4f}\n")
        f.write(f"F1 Score: {train_f1:.4f}\n\n")
        f.write("[Dev Set Evaluation - Isolation Forest]\n")
        f.write(f"AUC: {dev_auc:.4f}\n")
        f.write(f"Average Precision: {dev_ap:.4f}\n")
        f.write(f"F1 Score: {dev_f1:.4f}\n")
        f.write(f"Train Time: {train_time}\n")
        f.write(f"Predict Time: {pred_time}\n")

    plot_score_distribution(dev_scores, dev_labels, "Isolation Forest")
    return ('Isolation Forest', dev_auc, dev_ap, clf, None)

#tabular version of devnet
def devnet(train_features, train_labels, dev_features=None, dev_labels=None, retrain=False):
    print(f'using devnet for anomaly detection')
    output_file = "../output/devnet_results.txt"

    #ensure eveything is integers
    start_train = time.time()
    train_features = np.array(train_features).astype(int)
    train_labels =  np.array(train_labels).astype(int)
    if not retrain:
        dev_features = np.array(dev_features).astype(int)
        dev_labels = np.array(dev_labels).astype(int)

    #convert training labels to semi supervised format
    #1 = known anomaly, 0 = assumed normal
    semi_y = (train_labels == 1).astype(int)

    #initialize and train DevNet
    clf = DevNet(device='cpu')
    clf.fit(train_features, semi_y)
    train_time = time.time() - start_train

    if retrain:
        return('DevNet', None, None, clf, None)

    #train set evaluation
    start_pred = time.time()
    train_scores = clf.decision_function(train_features)
    pred_time = time.time() - start_pred
    train_auc, train_ap, train_f1 = tabular_metrics(train_labels, train_scores)

    with open(output_file, 'w') as f:
        f.write(f"\n[Train Set Evaluation]\n")
        f.write(f"AUC: {train_auc:.4f}\n")
        f.write(f"Average Precision: {train_ap:.4f}\n")
        f.write(f"F1 Score: {train_f1:.4f}\n")
        f.write(f"Train Time: {train_time}\n")
        f.write(f"Predict Time: {pred_time}\n")

    #dev set evaluation
    dev_scores = clf.decision_function(dev_features)
    dev_auc, dev_ap, dev_f1 = tabular_metrics(dev_labels, dev_scores)

    with open(output_file, 'a') as f:
        f.write(f"\n\n[Dev Set Evaluation]")
        f.write(f"AUC: {dev_auc:.4f}\n")
        f.write(f"Average Precision: {dev_ap:.4f}\n")
        f.write(f"F1 Score: {dev_f1:.4f}\n")
    
    plot_score_distribution(dev_scores, dev_labels, "DevNet")
    return ('DevNet', dev_auc, dev_ap, clf, None)


def cluster(train_features, train_labels, dev_features=None, dev_labels=None, retrain=False, best_k=None):
    print(f'using KMeans clustering for anomaly detection')
    output_file = "../output/kmeans_results.txt"

    if retrain:
        if best_k is None:
            best_k = 5
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        kmeans.fit(train_features)
        return ('KMeans', None, None, kmeans, best_k)

    best_k = None
    best_auc = 0
    best_kmeans = None

    # Try k from 2 to 10
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        start_train = time.time()
        kmeans.fit(train_features)
        train_time = time.time() - start_train

        # Anomaly score = distance to nearest cluster center
        train_distances = kmeans.transform(train_features).min(axis=1)
        train_distances = MinMaxScaler().fit_transform(train_distances.reshape(-1, 1)).flatten()

        auc = roc_auc_score(train_labels, train_distances)

        if auc > best_auc:
            best_auc = auc
            best_k = k
            best_kmeans = kmeans

    print(f"Best k = {best_k} (Train AUC = {best_auc:.4f})")

    # use best_kmeans to evaluate on train and dev sets
    # Train set eval
    train_distances = best_kmeans.transform(train_features).min(axis=1)
    train_distances = MinMaxScaler().fit_transform(train_distances.reshape(-1, 1)).flatten()
    train_auc = roc_auc_score(train_labels, train_distances)
    train_ap = average_precision_score(train_labels, train_distances)
    train_preds = train_distances > np.percentile(train_distances, 100 - 2.2)
    train_f1 = f1_score(train_labels, train_preds)

    # Dev set eval
    start_pred = time.time()
    dev_distances = best_kmeans.transform(dev_features).min(axis=1)
    dev_distances = MinMaxScaler().fit_transform(dev_distances.reshape(-1, 1)).flatten()
    predict_time = time.time() - start_pred
    dev_auc = roc_auc_score(dev_labels, dev_distances)
    dev_ap = average_precision_score(dev_labels, dev_distances)
    dev_preds = dev_distances > np.percentile(dev_distances, 100 - 2.2)
    dev_f1 = f1_score(dev_labels, dev_preds)

    with open(output_file, 'w') as f:
        f.write(f"Best k: {best_k}\n")
        f.write("[Train Set Evaluation - KMeans Clustering]\n")
        f.write(f"AUC: {train_auc:.4f}\n")
        f.write(f"Average Precision: {train_ap:.4f}\n")
        f.write(f"F1 Score: {train_f1:.4f}\n\n")
        f.write("[Dev Set Evaluation - KMeans Clustering]\n")
        f.write(f"AUC: {dev_auc:.4f}\n")
        f.write(f"Average Precision: {dev_ap:.4f}\n")
        f.write(f"F1 Score: {dev_f1:.4f}\n")
        f.write(f"Train Time: {train_time}\n")
        f.write(f"Predict Time: {predict_time}\n")
  
    plot_score_distribution(dev_distances, dev_labels, "KMeans")
    return ('KMeans', dev_auc, dev_ap, best_kmeans, best_k)


def plot_results(results):
    models = [r[0] for r in results]
    auc_scores = [r[1] for r in results]
    ap_scores = [r[2] for r in results]
    x = range(len(models))

    plt.figure(figsize=(10,6))
    plt.bar(x, auc_scores, width=0.4, label='AUC-ROC', align='center')
    plt.bar([i + 0.4 for i in x], ap_scores, width=0.4, label='AUC-PR', align='center')
    plt.xticks([i + 0.2 for i in x], models)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("AUC-ROC and AUC-PR Scores by Model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../output/auc_comparison.png")
    plt.show()


#function to predict on synthetic data to demonstrate useability of saved model
#function to predict on synthetic data to demonstrate useability of saved model
def predict_new_data(features, labels, model_path):
    output_file = "../output/artificial_data_results.txt"
    print(f"using the saved model to predict on the artificial data...")

    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        return

    # convert to int if needed (DevNet wants integer inputs)
    features = np.array(features).astype(int)
    labels = np.array(labels).astype(int)

    # validate feature dimension matches model
    expected_input_dim = None

    # validate feature dimension matches model
    expected_input_dim = None

    if hasattr(model, "input_dim"):  # DevNet
        expected_input_dim = model.input_dim
    elif hasattr(model, "n_features_in_"):  # Isolation Forest / KMeans
        expected_input_dim = model.n_features_in_

    if expected_input_dim is not None and features.shape[1] != expected_input_dim:
        print(f"Feature mismatch: model expects {expected_input_dim} features, but got {features.shape[1]}")
        return

    # predict based on model type
    if hasattr(model, "decision_function"):  # DevNet or IsolationForest
        scores = model.decision_function(features)
        scores = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).flatten()
    elif hasattr(model, "transform"):  # KMeans
        distances = model.transform(features).min(axis=1)
        scores = MinMaxScaler().fit_transform(distances.reshape(-1, 1)).flatten()
    else:
        print("[ERROR] Unknown model type.")
        return

    # evaluation
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    preds = scores >= np.percentile(scores, 100 - 2.2)
    f1 = f1_score(labels, preds)

    with open(output_file, "w") as f:
        f.write("\n[Evaluation on Artificial Data]\n")
        f.write(f"AUC-ROC: {auc:.4f}\n")
        f.write(f"Average Precision: {ap:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    # plot distribution
    plot_score_distribution(scores, labels, "Artificial_Data")

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["devnet", "iforest", "cluster", "all"], default="devnet")
parser.add_argument("--type", choices=["train_evaluation", "artificial"], default="train_evaluation")
args = parser.parse_args()

if __name__ == "__main__":
    check_dependencies() #ensure correct downloads

    if args.type == "train_evaluation":

        data = utils.read_data('../data/celeba_clean.csv') #load the cleaned version of the data (duplicate rows removed)

        #split the dataset into train and dev
        train_data, dev_data = train_test_split(data, test_size=0.2, random_state=42)

        #seperate features from labels
        train_features, train_labels = utils.extract_features(train_data, "class")
        dev_features, dev_labels = utils.extract_features(dev_data, "class")
        
        feature_importance(train_features, train_labels)

        results = [] 

        if args.model == "devnet":
            results.append(devnet(train_features, train_labels, dev_features, dev_labels))
        elif args.model == "iforest":
            results.append(iforest(train_features, train_labels, dev_features, dev_labels))
        elif args.model == "cluster":
            results.append(cluster(train_features, train_labels, dev_features, dev_labels))  # already returns 5 elements
        elif args.model == "baseline":
            results.append(baseline(train_features, train_labels, dev_features, dev_labels))
        elif args.model == "all":
            results.append(baseline(train_features, train_labels, dev_features, dev_labels))
            results.append(devnet(train_features, train_labels, dev_features, dev_labels))
            results.append(iforest(train_features, train_labels, dev_features, dev_labels))
            results.append(cluster(train_features, train_labels, dev_features, dev_labels))
            plot_results(results)

        #compare results and use the best model to retrain on entire data
        if results:
            best_model = max(results, key=lambda x: x[1]) #x[1] = AUC
            best_model_name = best_model[0]
            best_k_for_kmeans = best_model[4]

            print(f"Best Model: {best_model_name}")
            print(f"AUC-ROC: {best_model[1]:.4f}")
            print(f"Precision: {best_model[2]:.4f}")
            print(f"Retraining the best model on full data...")

        #separate features and labels on the full dataset
        features, labels = utils.extract_features(data, "class")
        
        #retrain the best model on the full clean data
        if best_model_name == 'DevNet':
            _, _, _, final_model, _ = devnet(features, labels, retrain=True)
        elif best_model_name == 'Isolation Forest':
            _, _, _, final_model, _ = iforest(features, labels, retrain=True)
        elif best_model_name == 'KMeans':
            _, _, _, final_model, _ = cluster(features, labels, retrain=True, best_k=best_k_for_kmeans)
        
        #save the fully trained best model
        joblib.dump(final_model, f"../output/{best_model_name.lower().replace(' ', '_')}_model.pkl")
    
    #sample script to demonstrate saved models useability on an artifically constructed test dataset
    elif args.type == "artificial":
        data = utils.read_data('../data/artificial_new_data.csv')

        #extract features and labels
        features, labels = utils.extract_features(data, "class")

        predict_new_data(features, labels, "../output/devnet_model.pkl")



    