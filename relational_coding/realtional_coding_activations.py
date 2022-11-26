import os

import pandas as pd

import config
from enums import Mode, Network
from mappings.re_arranging import rearrange_clips
from relational_coding.relational_coding_base import RelationalCoding
from statistical_analysis.math_functions import z_score
from statistical_analysis.matrices_ops import MatricesOperations
from supporting_functions import _dict_to_pkl


class RelationalCodingActivations(RelationalCoding):

    @staticmethod
    def _load_data(mode: Mode, network: Network):

        pass

    @classmethod
    def avg_single_tr_vectors(cls, net, sub_list, mode: Mode, clip: int, rest_tr: int = -1):
        clip_per_subs = []
        for sub in sub_list:
            if net == Network.WB:
                sub_matrix = pd.read_csv(os.path.join(config.ACTIVATION_MATRICES,
                                                      sub, mode.value, 'activation_matrix.csv'))
            else:
                sub_matrix = pd.read_csv(os.path.join(config.ACTIVATION_MATRICES,
                                                      sub, mode.value,
                                                      net.value,
                                                      'activation_matrix.csv'))

            activation_vec = cls.get_single_tr(sub_matrix, clip, 'tr', rest_tr)
            clip_per_subs.append(activation_vec)

        avg_series = MatricesOperations.get_avg_matrix((clip.values for clip in clip_per_subs))
        return avg_series

    @classmethod
    def compare_single_tr(cls, net, rest_tr):
        avg_series_per_clip = {}
        for clip_i, clip_name in config.idx_to_clip.items():
            if clip_name.startswith('test'):
                continue
            clip_vec = cls.avg_single_tr_vectors(net, config.sub_test_list, Mode.CLIPS, clip_name)
            rest_vec = cls.avg_single_tr_vectors(net, config.sub_test_list, Mode.REST_BETWEEN, clip_name, rest_tr)
            avg_series_per_clip[clip_name + '_' + Mode.CLIPS.value] = clip_vec.tolist()[0]
            avg_series_per_clip[clip_name + '_' + Mode.REST_BETWEEN.value] = rest_vec.tolist()[0]
        return pd.DataFrame.from_dict(avg_series_per_clip)

    @classmethod
    def corr_pipe_single_tr(cls, net, rest_tr, re_test: bool = False):
        _tr_mat = cls.compare_single_tr(net, rest_tr)
        _tr_mat = _tr_mat.apply(lambda x: z_score(x))
        _tr_corr = MatricesOperations.correlation_matrix(_tr_mat)
        _tr_corr = rearrange_clips(_tr_corr, where='rows', with_testretest=re_test)
        _tr_corr = rearrange_clips(_tr_corr, where='columns', with_testretest=re_test)
        return _tr_corr

    @classmethod
    def rest_tr_iteration(cls, net, re_test):
        relational_coding = {'correlation': [], 'relation_distance': []}

        for rest_tr in range(0, 19):
            tr_corr = cls.corr_pipe_single_tr(net, rest_tr, re_test)
            relational_coding['correlation'].append(tr_corr)
            relational_distance = cls.relational_distance(tr_corr.copy())
            relational_coding['relation_distance'].append(relational_distance)

        return relational_coding

    @classmethod
    def executor(cls, with_retest=False):
        relational_coding = {}
        for net in Network:
            print(net.name)
            relational_coding[net.name] = cls.rest_tr_iteration(
                net=net,
                re_test=with_retest)

        _dict_to_pkl(relational_coding, "Relational Distance LSTM patterns")

if __name__ == '__main__':

    relational_coding_instance = RelationalCodingActivations()
    relational_coding_instance.executor()