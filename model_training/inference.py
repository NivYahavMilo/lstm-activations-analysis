import os

import numpy as np
import pandas as pd
import torch

import config
from enums import Mode
from model_training.cc_utils import _lstm_test_acc
from model_training.dataloader import _get_clip_seq
from model_training.hyperparameters import HyperParams

def _test_time_window(df, window_range: range):
    


def _test(model: Mode, inference: Mode):
    test_subs = config.sub_test_list
    test_len = len(test_subs)

    args = HyperParams()
    model = torch.load(os.path.join(config.MODELS_PATH,
                                    f'{model.value}_model.pt'))
    args.device = torch.device('cpu')
    model.to(args.device)
    model.eval()

    rest_df = pd.read_pickle(os.path.join(config.FMRI_DATA,
                                          f'4_runs_{inference.value}.pkl'))

    X_test, X_len, y_test = _get_clip_seq(rest_df, test_subs, args)
    max_length = torch.max(X_len)
    '''
    results on test data
    '''
    # length of each clip
    clip_time = np.zeros(args.k_class)
    for ii in range(args.k_class):
        class_df = rest_df[rest_df['y'] == ii]
        clip_time[ii] = np.max(np.unique(class_df['timepoint'])) + 1
    clip_time = clip_time.astype(int)  # df saves float
    print('seq lengths = %s' % clip_time)

    a, a_t, c_mtx = _lstm_test_acc(model, X_test, y_test,
                                   X_len, max_length, clip_time,
                                   test_len, args)

    results = {'t_test': {}, 'test': a}
    for ii in range(args.k_class):
        results['t_test'][ii] = np.zeros(
            (test_len, clip_time[ii]))
    print('sacc = %0.3f' % np.mean(a))
    for ii in range(args.k_class):
        results['t_test'][ii] += a_t[ii]
    results['test_conf_mtx'] = c_mtx


if __name__ == '__main__':
    _test(model=Mode.CLIPS, inference=Mode.REST_BETWEEN)
