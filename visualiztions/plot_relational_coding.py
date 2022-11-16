import os

import numpy as np

import config
from enums import Network
from supporting_functions import _load_pkl
from sub_plot import plot_error_bar

def average_subjects_correlation(net: Network, path: str):
    subjects = config.sub_test_list
    subject_l = len(subjects)
    net_matrix = np.zeros((subject_l, 19))

    for ii,sub in enumerate(subjects):
        relational_coding_path_sub = os.path.join(path, sub, "Relational Distance fMRI.pkl")
        rc_sub = _load_pkl(relational_coding_path_sub)
        data = rc_sub[net.name]['relation_distance']
        net_matrix[ii, :] += data

    return net_matrix




def input_net_for_avg():
    relational_coding_path = os.path.join(config.ROOT_PATH, "experiments", "per_subject")
    r_c = {}
    for net in Network:
        r_c[net.name] = {}
        data_per_net = average_subjects_correlation(net, relational_coding_path)

        r_c[net.name]['mean'] = np.mean(data_per_net, axis=0)
        r_c[net.name]['std'] = np.std(data_per_net, axis=0)


    plot_error_bar(r_c)




if __name__ == '__main__':
    input_net_for_avg()