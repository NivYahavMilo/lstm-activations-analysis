import os

import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt

import config
from enums import Network
from statistical_analysis.matrices_ops import MatricesOperations
from supporting_functions import _load_pkl


def subplot_signal_from_dict(data):
    fig, ax = plt.subplots(len(data), 1, figsize=(7, 5))
    movies = data.keys()
    for a, key in zip(ax, movies):
        x = data[key]

        a.plot(list(x.keys()), list(x.values()))
    plt.title("Last tr clip Correlation with Resting state")
    plt.legend(movies)
    plt.xlabel("Rest TR")
    plt.ylabel("Correlation Value")

    plt.show()


def plot_signals_on_top(data, title):
    color_lst = ['black','blue','cyan','green',
                 'pink','red','violet','chocolate']
    correlations = []

    for _seq, _color in zip(data.items(), color_lst):
        clip, seq = _seq
        correlations.append({
            "name": clip,
            "x": list(seq["relation_distance"]),
            "Y": [1, -1],
            'color': colors.CSS4_COLORS[_color],
            'linewidth': 5,

        })

    fig, ax = plt.subplots(figsize=(20, 10), )

    for signal in correlations:
        ax.plot(signal['x'],  # signal['y'],
                color=signal['color'],
                linewidth=signal['linewidth'],
                label=signal['name'],
                )

    # Enable legend
    ax.legend(loc="upper right")
    ax.set_title(title)

    plt.xlabel("Rest TR")
    plt.ylabel("Correlation Value")
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(fr"{config.FIGURES_PATH}\\{title}.png", dpi=100)


def _subplot(table_path, title):
    df = pd.read_csv(table_path, index_col=0)
    plot_signals_on_top(data=df.to_dict(), title=title)


def _plot_wb():
    table_path = f'{config.CORRELATION_MATRIX_BY_TR}\\' \
                 f'fmri last tr clip correlation with rest between without test-re-test.csv'

    _subplot(table_path, "fMRI Last tr clip Correlation with Resting state")

    table_path = f'{config.CORRELATION_MATRIX_BY_TR}\\' \
                 f'last tr clip correlation with rest between without test-re-test.csv'
    title = "LSTM patterns similarity last tr clip with rest between"
    _subplot(table_path, title)


def _plot_networks(data_type):
    for net in Network:
        table_name = f"{data_type} {net.value}" \
                     f" last tr clip correlation with rest between without test-re-test"
        tables_path = os.path.join(
            config.CORRELATION_MATRIX_BY_TR,
            f"{table_name}.csv")
        _subplot(tables_path, table_name)


def _plot_combined_avg_wb_and_networks(data_type):
    combined_dict = {}
    wb_path = f'{config.CORRELATION_MATRIX_BY_TR}\\' \
              f'{data_type} WB last tr clip correlation with rest between without test-re-test.csv'
    wb_df = pd.read_csv(wb_path, index_col=0)
    for ii, movie in wb_df.iterrows():
        avg_vec = round(MatricesOperations.get_avg_vector(movie), 3)
        combined_dict.setdefault("WB", {}).update({ii: avg_vec})

    for net in Network:
        table_name = f"{data_type} {net.value}" \
                     f" last tr clip correlation with rest between without test-re-test"
        net_path = os.path.join(
            config.CORRELATION_MATRIX_BY_TR,
            f"{table_name}.csv")

        net_df = pd.read_csv(net_path, index_col=0)
        for ii, movie in net_df.iterrows():
            avg_vec = round(MatricesOperations.get_avg_vector(movie), 3)
            combined_dict.setdefault(net.name, {}).update({ii: avg_vec})

    plot_signals_on_top(
        combined_dict,
        'Average movies correlation with rest between')


def plot_relation_coding_fmri():
    data = _load_pkl("Relational Distance fMRI.pkl")
    plot_signals_on_top(data, title="Relation Coding fMRI")




if __name__ == '__main__':
    # _plot_networks(data_type="lstm patterns")
    # _plot_networks(data_type="fmri")
    # _plot_combined_avg_wb_and_networks("lstm patterns")
    # _plot_combined_avg_wb_and_networks("fmri")
    plot_relation_coding_fmri()
