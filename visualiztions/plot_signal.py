import os

import numpy as np
import pandas as pd
import config
import matplotlib.pyplot as plt
from statistical_analysis.matrices_ops import MatricesOperations

K_RUNS = 4


def load_fmri_data(file, net=None):
    if net:
        df = pd.read_pickle(os.path.join(config.FMRI_DATA, 'networks_df', f'{file}.pkl'))
    else:
        df = pd.read_pickle(os.path.join(config.FMRI_DATA, f'{file}.pkl'))
    return df


def filter_wanted_signal(data, subject, i_class):
    features = [feat for feat in data.columns if feat.startswith('feat')]

    if i_class == 0:  # split test-retest into 4
        test_retest = {}
        seqs = data[(data['Subject'] == subject) &
                    (data['y'] == 0)][features].values
        k_time = int(seqs.shape[0] / K_RUNS)
        for i_run in range(K_RUNS):
            seq = seqs[i_run * k_time:(i_run + 1) * k_time, :]
            test_retest[i_run] = seq

        return test_retest

    else:
        seq = data[(data['Subject'] == subject) &
                   (data['y'] == i_class)][features].values

        return seq


def avg_roi_seq(signals, clip_i):
    one_dimension_sig = {}
    if clip_i == 0:

        for clip_run, seq in signals.items():
            signal = f'testretest{str(clip_run + 1)}'
            # average roi
            roi_avg = MatricesOperations.get_avg_matrix(iter(seq), axis=1)
            one_dimension_sig[signal] = roi_avg
    else:
        # average roi
        signal = config.idx_to_clip.get(clip_i)
        roi_avg = MatricesOperations.get_avg_matrix(iter(signals), axis=1)
        one_dimension_sig[signal] = roi_avg

    return one_dimension_sig


def iterate_subs(clip_i, mode):
    df_clips = load_fmri_data('4_movie_runs')
    subjects_signals = {}
    for subject in config.sub_test_list:
        subject = int(subject)
        seq_signals = filter_wanted_signal(df_clips, subject, clip_i)

        subjects_signals[subject] = avg_roi_seq(seq_signals, clip_i)

    average_signal = average_all_subjects(subjects_signals,
                                          clip_i,
                         test_retest=True if clip_i == 0 else False,
                         concat_test_retest=False)
    return average_signal


def average_all_subjects(subjects_signals, clip_i,  test_retest: bool = False, concat_test_retest: bool = False):
    avg_seq = {}
    clip = config.connectivity_mapping.get(clip_i)
    if test_retest:
        all_subjects = {}
        for sub, seqs in subjects_signals.items():
            for clip, seq in seqs.items():
                all_subjects.setdefault(clip, []).append(seq)

        for clip, lst_seq in all_subjects.items():
            avg_seq[clip] = MatricesOperations.get_avg_matrix(iter(lst_seq))

        if concat_test_retest:
            concat_seq = np.concatenate([v for k,v in avg_seq.items()])
            return concat_seq

    else:
        avg_seq = MatricesOperations.get_avg_matrix(iter([seqs.get(clip) for sub, seqs in subjects_signals.items()]))

    return avg_seq


def plot_signal(signal, clip):
    if len(signal) == 4:
        mysignals = []
        c = 0
        colors = ['b', 'r', 'g', 'k']
        for clip, seq in signal.items():
            mysignals.append({
                "name": clip,
                "x": seq,
                'color': colors[c],
                'linewidth': 1

            })
            c+=1
        fig, ax = plt.subplots()
        for signal in mysignals:
            ax.plot(signal['x'], # signal['y'],
                    color=signal['color'],
                    linewidth=signal['linewidth'],
                    label=signal['name'])

        # Enable legend
        ax.legend()
        ax.set_title("test-re-test")
        plt.xlabel("TR")
        plt.ylabel("BOLD")
        plt.show()
    else:

        plt.plot(signal)
        plt.xlabel("TR")
        plt.ylabel("BOLD")
        plt.title(clip)
        plt.show()

if __name__ == '__main__':
    inception = 5
    avg_signal = iterate_subs(clip_i=inception, mode=None)
    clip_name = config.idx_to_clip.get(inception)
    plot_signal(avg_signal, clip_name)

    pockets = 3
    avg_signal = iterate_subs(clip_i=pockets, mode=None)
    clip_name = config.idx_to_clip.get(pockets)
    plot_signal(avg_signal, clip_name)

    testretest = 0
    avg_signal = iterate_subs(clip_i=testretest, mode=None)
    clip_name = config.idx_to_clip.get(testretest)
    plot_signal(avg_signal, clip_name)
    pass
