import pandas as pd

#Authors: Aubrie Pressley & Lisette Kamper Hinson

#function to read the data from a csv file and store in a dataframe
def read_data(file_path, encoding='utf-8'):
    data = pd.read_csv(file_path, header=0)
    return data

#function to extract features and labels
def extract_features(data, label_col):
    labels = data[label_col]
    features = data.drop([label_col], axis=1, errors='ignore')
    return features, labels