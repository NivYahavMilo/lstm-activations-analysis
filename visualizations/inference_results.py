import numpy as np
import pandas as pd
import pickle
import os

from enums import Mode
from model_training.cc_utils import _get_clip_labels

# plot
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots

import config

pio.templates.default = 'plotly_white'
from plot_utils import _hex_to_rgb, _plot_ts

# colors = px.colors.qualitative.Plotly
colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf',  # blue-teal
    '#0d0887',
    '#46039f',
    '#7201a8',
    '#9c179e',
    '#bd3786']

class ARGS:
    roi = 300
    net = 7
    subnet = 'wb'
    zscore = 1
    k_fold = 10
    k_hidden = 150
    batch_size = 16
    num_epochs = 50
    train_size = 100

    # lstm
    lstm_layers = 1

    # tcn
    tcn_wind = 30

    # ff
    ff_layers = 5


args = ARGS()


def _get_results(args, model, networks=None):
    res_path = config.RESULTS_PATH if not networks else config.RESULTS_PATH_NETWORKS
    load_path = os.path.join(res_path,model)
    with open(load_path, 'rb') as f:
        r = pickle.load(f)
    return r


def _compare_mean(models, mode='train_mode'):
    '''
    compare mean accuracy for each model
    '''
    fig = go.Figure()
    ticktext = []
    ii = 0
    for model, file in models.items():
        args.lstm_layers = 1
        ticktext.append(model)

        r = _get_results(args, file)['test_mode']
        if mode == 'train_mode':
            tag = 'val'
            multiplier = 3
        elif mode == 'test_mode':
            tag = 'test'
            multiplier = 1

        y = np.mean(r[tag])
        err_y = 1 / np.sqrt(args.k_fold) * np.std(r[tag])

        bar = go.Bar(x=[ii], y=[y],
                     error_y=dict(type='data',
                                  array=[multiplier * err_y], width=9),
                     name=model,
                     showlegend=True,
                     marker_color=colors[ii])
        fig.add_trace(bar)
        ii+=1
    fig.update_yaxes(range=[0, 1],
                     title=dict(text='Clip Prediction Accuracy',
                                font_size=25),
                     gridwidth=1, gridcolor='#bfbfbf',
                     tickfont=dict(size=20))
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(height=500, width=450,
                      font_color='black',
                      legend_orientation='h',
                      legend_font_size=20,
                      legend_x=-0.1)

    return fig


def _compare_temporal(models, clip_names, mode='train_mode'):
    '''
    compare temporal accuracy for each model
    '''

    k_class = len(clip_names)
    k_rows = int(np.ceil(k_class / 3))
    k_cols = 3
    fig = make_subplots(rows=k_rows, cols=k_cols,
                        subplot_titles=clip_names, print_grid=False)

    fig_clip = {}
    for clip in clip_names:
        fig_clip[clip] = go.Figure()
    ii=0
    for model, file in models.items():
        args.lstm_layers = 1

        r = _get_results(args, file)['test_mode']
        if mode == 'train_mode':
            tag = 't_val'
        elif mode == 'test_mode':
            tag = 't_test'
        name = model

        max_time = -100
        for jj in range(k_class):
            row = int(jj / k_cols) + 1
            col = (jj % k_cols) + 1

            showlegend = False
            if jj == 0:
                showlegend = True

            acc = r[tag][jj]
            ts = {'mean': np.mean(acc, axis=0),
                  'ste': 1 / np.sqrt(len(acc)) * np.std(acc, axis=0)}

            plotter = _plot_ts(ts, colors[ii],
                               showlegend=showlegend, name=name)
            for trace in plotter:
                fig.add_trace(plotter[trace], row, col)
            for trace in plotter:
                fig_clip[clip_names[jj]].add_trace(plotter[trace])

            if len(ts['mean']) > max_time:
                max_time = len(ts['mean'])
        ii+=1

    fig.update_layout(height=int(250 * k_rows), width=750,
                      legend_orientation='h')
    fig.update_xaxes(range=[0, max_time], dtick=30,
                     title_text='time (in s)',
                     showgrid=False,
                     autorange=False)
    fig.update_yaxes(range=[0, 1], dtick=0.2,
                     gridwidth=1, gridcolor='#bfbfbf',
                     autorange=False)

    return fig, fig_clip

def _get_labels():

    timing_file = pd.read_csv(os.path.join(config.FMRI_DATA,
                              f'videoclip_tr_lookup.csv'))

    clip_y = _get_clip_labels(timing_file)
    k_class = len(np.unique(list(clip_y.values())))
    print('number of classes = %d' %k_class)

    clip_names = np.zeros(k_class).astype(str)
    clip_names[0] = 'testretest'
    for key, item in clip_y.items():
        if item!=0:
            clip_names[item] = key

    return clip_names


def mean_pipeline(models):

    fig = _compare_mean(models, mode='test_mode')
    fig.show()
    fig_mean = go.Figure(fig)

def temporal_pipeline(models):
    clip_names = _get_labels()

    fig, fig_clip = _compare_temporal(models, clip_names,
                                      mode='test_mode')
    fig.show()


def main():
    # models = {'first_10_tr': '300 roi rest_between 0-10 tr results.pkl',
    #           'last_10_tr': '300 roi rest_between 10-20 tr results.pkl'}

    models = {'combined': '300 roi Mode.COMBINED results.pkl'}
    mean_pipeline(models)
    temporal_pipeline(models)

if __name__ == '__main__':
    main()