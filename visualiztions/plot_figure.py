import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from enums import Network, DataType
from supporting_functions import _load_pkl


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


def plot_heat_activations(data, data_type: DataType):
    for net, values in data.items():
        network_data = data[net]
        correlation_maps = network_data['correlation']
        if net == Network.Default.name or net == Network.Visual.name:
            for tr in [2, 10, 18]:
                title = f"{data_type.value} Rational Coding Correlation {net} at {tr} TR"
                plot_matrix(correlation_maps[tr],
                            title=title)
                print(title)




if __name__ == '__main__':
    data_ = _load_pkl("Relational Distance LSTM patterns.pkl")
    plot_heat_activations(data_, data_type=DataType.LSTM_PATTERNS)
    data_ = _load_pkl("Relational Distance fMRI.pkl")
    plot_heat_activations(data_, data_type=DataType.FMRI)
