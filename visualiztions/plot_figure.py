import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def save_matrix_as_image():
    SMALL_SIZE = 4
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    mat_path = os.path.join(
        config.DATA_PATH, 'activations_matrices', 'avrage_accross_all_subjects')
    for array in os.listdir(f'{mat_path}\\csvis'):
        if not array.isdigit():
            continue
        array_ = pd.read_csv(f'{mat_path}\\csvis\\{array}').values
        # fontsize of the x and y labels
        plt.rcParams.update(
            {'font.size': 4, 'xtick.labelsize': 2, 'ytick.labelsize': 6,
             'figure.max_open_warning': 40})
        plt.figure(figsize=(40, 30))
        plt.pcolor(array_, cmap='hot')
        plt.colorbar()
        plt.savefig(
            os.path.join(mat_path, 'images', array.replace('.csv', '.png')))


def plot_matrix(matrix: pd.DataFrame, title, cmap: str = 'hot'):
    SMALL_SIZE = 4
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)
    # fontsize of the x and y labels
    plt.rcParams.update(
        {'font.size': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6})
    plt.figure(figsize=(10, 6))
    plt.pcolor(matrix, cmap=cmap)
    plt.colorbar()
    plt.title(title, fontdict={"fontsize": "16"})
    labels = [lab.replace('_between', '') for lab in matrix.columns if
              isinstance(lab, str)]
    plt.yticks(np.arange(0.5, len(matrix.index), 1), labels)
    plt.xticks(np.arange(0.5, len(matrix.columns), 1), labels, rotation=70)
    plt.show()
    print()


if __name__ == '__main__':
    mat = pd.read_csv('average_correlation_matrix.csv',
                      index_col=0)
    mat1 = pd.read_csv('avg_connectivity_300roi.csv',
                       index_col=0)

    plot_matrix(mat, title="activations_300roi")
    plot_matrix(mat1, title="connectivity_300roi")
