import utils
import numpy as np
import pandas as pd
import os

#script to perform necessary data cleaning on the rows
#save the clean data to a csv file in the data directory
#according to the quality report, the only necessary cleaning is removing duplicate rows
#script only need to be performed once in the beginning of the project

#Authors: Aubrie Pressley & Lisette Kamper Hinson

#function to remove duplicate rows
def remove_duplicate(data):
    data = pd.DataFrame(data) #insure its a dataframe
    return data.drop_duplicates()

# function to convert all columns to integers, forcing any invalid to NaN
def convert_num(data):
    data = pd.DataFrame(data)

    # strip whitespace (just in case) and convert everything to int
    for col in data.columns:
        data[col] = data[col].astype(str).str.strip()  # remove any hidden spaces
        data[col] = pd.to_numeric(data[col], errors='coerce')  # convert to numeric

    if data.isnull().any().any():
        print("warning: some values could not be converted to integers.")
        print(data.isnull().sum())

    # final conversion to int
    return data.astype(int)


#function to save the dataframe as a csv file
def save_csv(file_path, data):
    data.to_csv(file_path, index=False)

def main():
    data = utils.read_data("../data/celeba_baldvsnonbald_normalised.csv")
    data = remove_duplicate(data)
    data = convert_num(data) #convert all values to int
    save_csv('../data/celeba_clean.csv', data)

if __name__ == "__main__":
    main()
