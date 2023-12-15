import numpy as np
import pandas as pd

def load_dataset():
    # Load the CSV files using Pandas
    train_dataset = pd.read_csv('train_labels.csv')
    test_dataset = pd.read_csv('test_labels.csv')

    # Extract features (file paths) and labels
    train_set_x_orig = np.array(train_dataset['Filepath'])
    train_set_y_orig = np.array(train_dataset['Label'])

    test_set_x_orig = np.array(test_dataset['Filepath'])
    test_set_y_orig = np.array(test_dataset['Label'])

    # Extract classes
    classes = np.unique(train_set_y_orig)

    # Reshape labels
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


