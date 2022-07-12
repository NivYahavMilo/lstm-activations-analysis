import os
import pickle

import numpy as np
import pandas as pd
import torch

import config
from enums import Mode
from model_training.cc_utils import _lstm_test_acc, _test_time_window
from model_training.dataloader import _get_clip_seq
from model_training.hyperparameters import HyperParams


def _test(model_mode: Mode, inference: Mode, tr_range: tuple = ()):
    test_subs = config.sub_test_list
    test_len = len(test_subs)

    args = HyperParams()
    model = torch.load(os.path.join(config.MODELS_PATH,
                                    f'{model_mode.value}_model.pt'))
    args.device = torch.device('cpu')
    model.to(args.device)
    model.eval()

    clip_df = pd.read_pickle(os.path.join(config.FMRI_DATA,
                                          f'4_runs_{inference.value}.pkl'))
    if tr_range:
        start_tr, end_tr = tr_range
        clip_df = _test_time_window(clip_df,
                                    range(start_tr, end_tr))

    X_test, X_len, y_test = _get_clip_seq(clip_df, test_subs, args)
    max_length = torch.max(X_len)
    '''
    results on test data
    '''
    # length of each clip
    clip_time = np.zeros(args.k_class)
    for ii in range(args.k_class):
        class_df = clip_df[clip_df['y'] == ii]
        clip_time[ii] = np.max(np.unique(class_df['timepoint'])) + 1
    clip_time = clip_time.astype(int)  # df saves float
    print('seq lengths = %s' % clip_time)

    a, a_t, c_mtx = _lstm_test_acc(
        model=model,
        X=X_test,
        y=y_test,
        X_len=X_len,
        max_length=max_length,
        clip_time=clip_time,
        k_sub=test_len,
        args=args,
        save_activations=True)

    results = {'t_test': {}, 'test': a}
    for ii in range(args.k_class):
        results['t_test'][ii] = np.zeros(
            (test_len, clip_time[ii]))
    print('sacc = %0.3f' % np.mean(a))
    for ii in range(args.k_class):
        results['t_test'][ii] += a_t[ii]
    results['test_conf_mtx'] = c_mtx

    res_path = f'pred-{inference}_{model_mode.value}{tr_range[0]}-{tr_range[1]}' \
        if tr_range else f'pred-{inference.value}_model_{model_mode.value}'
    return results, res_path

def _save(results: dict, path: str):
    results_ = {"test_mode":  results}

    with open(os.path.join(config.RESULTS_PATH,
                           f'{path}.pkl'),
              'wb') as f:
        pickle.dump(results_, f)


if __name__ == '__main__':

    test_res, res_path = _test(
        model_mode=Mode.COMBINED,
        inference=Mode.REST_BETWEEN)
    _save(test_res, res_path)
    # _test(model=Mode.COMBINED, inference=Mode.COMBINED, tr_range=(10, 20))
