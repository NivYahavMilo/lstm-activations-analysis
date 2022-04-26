
import numpy as np
import pandas as pd
import pickle
import os
import time

import config
from supporting_functions import _dict_to_pkl
import torch
import torch.nn as nn
from models import LSTMClassifier

from utils import _info
from cc_utils import _lstm_test_acc
from dataloader import _get_clip_seq as _get_seq
from dataloader import _clip_class_df
from torch_utils import getActivation, activation

K_SEED = 330

def _test(df, args):
    '''
    test subject results
    view only for best cross-val parameters
    '''
    _info('test mode')

    # set pytorch device
    torch.manual_seed(K_SEED)
    use_cuda = torch.cuda.is_available()
    args.device = torch.device('cuda:0' if use_cuda else 'cpu')
    if use_cuda:
        _info('cuda')
    else:
        _info('cpu')

    # get X-y from df
    subject_list = df['Subject'].unique()
    train_list = subject_list[:args.train_size]
    test_list = subject_list[args.train_size:]

    features = [ii for ii in df.columns if 'feat' in ii]
    k_feat = len(features)
    print('number of classes = %d' % (args.k_class))

    # length of each clip
    clip_time = np.zeros(args.k_class)
    for ii in range(args.k_class):
        class_df = df[df['y'] == ii]
        clip_time[ii] = np.max(np.unique(class_df['timepoint'])) + 1
    clip_time = clip_time.astype(int)  # df saves float
    _info('seq lengths = %s' % clip_time)

    # results dict init
    results = {}

    # mean accuracy across time
    results['train'] = np.zeros(len(test_list))
    results['val'] = np.zeros(len(test_list))

    # per class temporal accuracy
    results['t_train'] = {}
    results['t_test'] = {}
    for ii in range(args.k_class):
        results['t_train'][ii] = np.zeros(
            (len(test_list), clip_time[ii]))
        results['t_test'][ii] = np.zeros(
            (len(test_list), clip_time[ii]))
    '''
    init model
    '''
    model = LSTMClassifier(k_feat, args.k_hidden,
                           args.k_layers, args.k_class)
    model.to(args.device)
    print(model)

    lossfn = nn.CrossEntropyLoss(ignore_index=-100)
    # if input is cuda, loss function is auto cuda
    opt = torch.optim.Adam(model.parameters())

    # get train, val sequences
    X_train, train_len, y_train = _get_seq(df,
                                           train_list, args)
    X_test, test_len, y_test = _get_seq(df,
                                        test_list, args)

    max_length = torch.max(train_len)
    permutation = torch.randperm(X_train.size()[0])
    losses = np.zeros(args.num_epochs)
    #
    then = time.time()
    if not args.inference:
        for epoch in range(args.num_epochs):
            for i in range(0, X_train.size()[0], args.batch_size):
                indices = permutation[i:i + args.batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]
                batch_x_len = train_len[indices]

                y_pred = model(batch_x, batch_x_len, max_length)
                loss = lossfn(y_pred.view(-1, args.k_class),
                              batch_y.view(-1))

                opt.zero_grad()
                loss.backward()
                opt.step()

    torch.save(model, os.path.join(config.MODELS_PATH, 'rest_between_model.pt'))
    # torch.save(model.state_dict(), 'model_state_dict.pt')

    if args.inference:
        model = torch.load('model.pt')
    '''
    results on test data
    '''
    a, a_t, c_mtx = _lstm_test_acc(model, X_test, y_test,
                                   test_len, max_length, clip_time, len(test_list), return_states=False)

    _dict_to_pkl(activation, 'rest_between_model_activations')


def main(args):

    res_path = (RES_DIR +
                '/roi_%d_net_%d' % (args.roi, args.net) +
                '_nw_%s' % (args.subnet) +
                '_trainsize_%d' % (args.train_size) +
                '_kfold_%d_k_hidden_%d' % (args.k_fold, args.k_hidden) +
                '_k_layers_%d_batch_size_%d' % (args.k_layers, args.batch_size) +
                '_num_epochs_%d_z_%d_rest_between.pkl' % (args.num_epochs, args.zscore))
    if not os.path.isfile(res_path):
        df = _clip_class_df(args)
        df.to_pickle(os.path.join(config.DATA_PATH,'4_runs_rest_between.pkl'))
    results = {}
    df = pd.read_pickle(os.path.join(config.FMRI_DATA, '4_runs_rest_between.pkl'))
    results['test_mode'] = _test(df, args)
    with open(res_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    pass
