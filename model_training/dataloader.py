import os.path

import numpy as np
import pandas as pd
import random
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from model_training.cc_utils import _get_clip_labels
from model_training.fmri_utils import _get_parcel

import config
from enums import Mode

K_RUNS = 4
K_SEED = 330


def _get_clip_seq(df, subject_list, args):
    '''
    return:
    X: input seq (batch_size x time x feat_size)
    y: label seq (batch_size x time)
    X_len: len of each seq (batch_size x 1)
    batch_size <-> number of sequences
    time <-> max length after padding
    '''
    features = [ii for ii in df.columns if 'feat' in ii]

    X = []
    y = []
    for subject in subject_list:
        for i_class in range(args.k_class):

            if i_class == 0:  # split test-retest into 4
                seqs = df[(df['Subject'] == int(subject)) &
                          (df['y'] == 0)][features].values
                label_seqs = df[(df['Subject'] == int(subject)) &
                                (df['y'] == 0)]['y'].values

                k_time = int(seqs.shape[0] / K_RUNS)
                for i_run in range(K_RUNS):
                    seq = seqs[i_run * k_time:(i_run + 1) * k_time, :]
                    label_seq = label_seqs[i_run * k_time:(i_run + 1) * k_time]
                    if args.zscore:
                        # zscore each seq that goes into model
                        seq = (1 / np.std(seq)) * (seq - np.mean(seq))

                    X.append(torch.FloatTensor(seq))
                    y.append(torch.LongTensor(label_seq))
            else:
                seq = df[(df['Subject'] == int(subject)) &
                         (df['y'] == i_class)][features].values
                label_seq = df[(df['Subject'] == int(subject)) &
                               (df['y'] == i_class)]['y'].values
                if args.zscore:
                    # zscore each seq that goes into model
                    seq = (1 / np.std(seq)) * (seq - np.mean(seq))

                X.append(torch.FloatTensor(seq))
                y.append(torch.LongTensor(label_seq))

    X_len = torch.LongTensor([len(seq) for seq in X])

    # pad sequences
    X = pad_sequence(X, batch_first=True, padding_value=0)
    y = pad_sequence(y, batch_first=True, padding_value=-100)

    return X.to(args.device), X_len.to(args.device), y.to(args.device)


def _clip_class_df(args):
    '''
    data for 15-way clip classification
    args.roi: number of ROIs
    args.net: number of subnetworks (7 or 17)
    args.subnet: subnetwork; 'wb' if all subnetworks
    args.invert_flag: all-but-one subnetwork
    args.r_roi: number of random ROIs to pick
    args.r_seed: random seed for picking ROIs
    save each timepoint as feature vector
    append class label based on clip
    return:
    pandas df
    '''
    # optional arguments
    d = vars(args)
    if 'invert_flag' not in d:
        args.invert_flag = False
    if 'r_roi' not in d:
        args.r_roi = 0
        args.r_seed = 0

    load_path = (os.path.join(config.FMRI_DATA,
                              'data_MOVIE_runs_roi_300_net_7_ts.pkl'))

    with open(load_path, 'rb') as f:
        data = pickle.load(f)

    # where are the clips within the run?
    mode_mapping = {Mode.CLIPS: 'videoclip_tr_lookup',
                    Mode.REST_BETWEEN: 'restclip_tr_lookup',
                    Mode.COMBINED: 'combined_clip_tr_lookup'}

    timing_file = pd.read_csv(os.path.join(config.FMRI_DATA,
                              f'{mode_mapping.get(args.mode)}.csv'))

    # pick either all ROIs or subnetworks
    if args.subnet != 'wb':
        if 'minus' in args.subnet:
            # remove 'minus_' prefix
            args.subnet = args.subnet.split('minus_')[1]

        _, nw_info = _get_parcel(args.roi, args.net)
        # ***roi ts sorted in preprocessing
        nw_info = np.sort(nw_info)
        idx = (nw_info == args.subnet)
    else:
        idx = np.ones(args.roi).astype(bool)

    # all-but-one subnetwork
    if args.subnet and args.invert_flag:
        idx = ~idx

    # if random selection,
    # overwrite everything above
    if args.r_roi > 0:
        random.seed(args.r_seed)
        idx = np.zeros(args.roi).astype(bool)
        # random sample without replacement
        samp = random.sample(range(args.roi), k=args.r_roi)
        idx[samp] = True
    '''
    main
    '''
    clip_y = _get_clip_labels(timing_file)

    table = []
    for run in range(K_RUNS):

        print('loading run %d/%d' % (run + 1, K_RUNS))
        run_name = 'MOVIE%d' % (run + 1)  # MOVIEx_7T_yz

        # timing file for run
        timing_df = timing_file[
            timing_file['run'].str.contains(run_name)]
        timing_df = timing_df.reset_index(drop=True)

        for subject in data:

            # get subject data (time x roi x run)
            roi_ts = data[subject][:, idx, run]

            for jj, clip in timing_df.iterrows():

                start = int(np.floor(clip['start_tr']))
                stop = int(np.ceil(clip['stop_tr']))
                clip_length = stop - start

                # assign label to clip
                y = clip_y[clip['clip_name']]

                for t in range(clip_length):
                    act = roi_ts[t + start, :]
                    t_data = {}
                    t_data['Subject'] = subject
                    t_data['timepoint'] = t
                    for feat in range(roi_ts.shape[1]):
                        t_data['feat_%d' % (feat)] = act[feat]
                    t_data['y'] = y
                    table.append(t_data)

    df = pd.DataFrame(table)
    df['Subject'] = df['Subject'].astype(int)

    return df