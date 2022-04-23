import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

import config


def _load_csv(path):
    return pd.read_csv(path, index_col=0)


def save_matrix_as_image():
    SMALL_SIZE = 4
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    mat_path = os.path.join(
        config.DATA_PATH, 'activations_matrices', 'avrage_accross_all_subjects')
    for array in os.listdir(f'{mat_path}\\csvis'):
        if array == 'average_correlation_matrix.csv' or array == 'avrage_across_all_subjects.zip':
            continue
        array_ = pd.read_csv(f'{mat_path}\\csvis\\{array}').values
        # fontsize of the x and y labels
        plt.rcParams.update({'font.size': 4, 'xtick.labelsize': 2, 'ytick.labelsize': 6, 'figure.max_open_warning':40})
        plt.figure(figsize=(40, 30))
        plt.pcolor(array_, cmap='hot')
        plt.colorbar()
        plt.savefig(os.path.join(mat_path, 'images', array.replace('.csv', '.png')))


def plot_matrix(matrix: pd.DataFrame, cmap: str = 'hot'):
    SMALL_SIZE = 4
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)
    # fontsize of the x and y labels
    plt.rcParams.update({'font.size': 4, 'xtick.labelsize': 2, 'ytick.labelsize': 6})
    plt.figure(figsize=(10, 6))
    plt.pcolor(matrix, cmap=cmap)
    plt.colorbar()
    labels = [lab.replace('_between', '') for lab in matrix.columns]
    plt.yticks(np.arange(0.5, len(matrix.index), 1), labels)
    plt.xticks(np.arange(0.5, len(matrix.columns), 1), labels)
    plt.show()
    print()


if __name__ == '__main__':
    save_matrix_as_image()
