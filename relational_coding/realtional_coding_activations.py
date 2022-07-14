import os

import pandas as pd

import config
from enums import Mode, Network
from mappings.re_arranging import rearrange_clips
from relational_coding.relational_coding_base import RelationalCoding
from statistical_analysis.math_functions import z_score
from statistical_analysis.matrices_ops import MatricesOperations


class RelationalCodingActivations(RelationalCoding):

    @staticmethod
    def _load_data(mode: Mode, network: Network):
        pass

    @classmethod
    def avg_single_tr_vectors(cls, sub_list, mode: Mode, clip: int, rest_tr: int):
        clip_per_subs = []
        for sub in sub_list:
            sub_matrix = pd.read_csv(os.path.join(config.ACTIVATION_MATRICES,
                                                  sub, mode.value, 'activation_matrix.csv'))

            activation_vec = cls.get_single_tr(sub_matrix, clip, 'tr', rest_tr)
            clip_per_subs.append(activation_vec)
        avg_series = MatricesOperations.get_avg_matrix((clip.values for clip in clip_per_subs))
        return avg_series

    @classmethod
    def compare_single_tr(cls, rest_tr):
        avg_series_per_clip = {}
        for clip_i, clip_name in config.idx_to_clip():
            clip_vec = cls.avg_single_tr_vectors(config.sub_test_list, Mode.CLIPS, clip_i, rest_tr)
            rest_vec = cls.avg_single_tr_vectors(config.sub_test_list, Mode.REST_BETWEEN, clip_i, rest_tr)
            avg_series_per_clip[clip_name + '_' + Mode.CLIPS.value] = clip_vec.tolist()[0]
            avg_series_per_clip[clip_name + '_' + Mode.REST_BETWEEN.value] = rest_vec.tolist()[0]
        return pd.DataFrame.from_dict(avg_series_per_clip)

    @classmethod
    def corr_pipe_single_tr(cls, table_name, re_test: bool = False):
        _tr_mat = cls.compare_cliptorest_single_tr()
        _tr_mat = _tr_mat.apply(lambda x: z_score(x))
        _tr_corr = MatricesOperations.auto_correlation_matrix(_tr_mat)
        _tr_corr = rearrange_clips(_tr_corr, where='rows', with_testretest=re_test)
        _tr_corr = rearrange_clips(_tr_corr, where='columns', with_testretest=re_test)
