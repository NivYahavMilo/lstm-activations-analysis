import os

import pandas as pd

import config
from enums import Network, Mode
from mappings.re_arranging import rearrange_clips
from relational_coding.relational_coding_base import RelationalCoding
from statistical_analysis.math_functions import z_score
from statistical_analysis.matrices_ops import MatricesOperations
from supporting_functions import _dict_to_pkl


class RelationalCodingfMRI(RelationalCoding):

    @staticmethod
    def _load_data(mode: Mode, network: Network):
        net_path = os.path.join(config.FMRI_DATA_NETWORKS,
                                mode.value,
                                f'df{network.value}.pkl')

        if network == Network.WB:
            net_path = os.path.join(config.FMRI_DATA,
                                f'4_runs_{mode.value}.pkl')

        data = pd.read_pickle(net_path)
        return data

    @classmethod
    def avg_single_tr_vectors(cls, data, sub_list, clip_index, tr_pos):
        clip_per_subs = []
        for sub in sub_list:
            sub_matrix = data[(data['Subject'] == int(sub)) &
                              (data['y'] == clip_index)]
            fmri_vec = cls.get_single_tr(mat_clip=sub_matrix,
                                         clip_i=clip_index,
                                         tr_field='timepoint',
                                         tr_pos=tr_pos)
            clip_per_subs.append(fmri_vec)
        avg_series = MatricesOperations.get_avg_matrix(
            (clip.values for clip in clip_per_subs))
        return avg_series

    @classmethod
    def compare_clip_to_rest_single_tr(cls, movies, rest, rest_tr):
        avg_series_per_clip = {}
        for clip_i, clip in config.connectivity_mapping.items():
            if clip_i == 0:
                continue

            clip_vec = cls.avg_single_tr_vectors(
                data=movies,
                sub_list=config.sub_test_list,
                tr_pos=-1,
                clip_index=clip_i)

            avg_series_per_clip[
                clip + '_' + Mode.CLIPS.value] = clip_vec.tolist()[0]

            rest_vec = cls.avg_single_tr_vectors(
                data=rest,
                sub_list=config.sub_test_list,
                tr_pos=rest_tr,
                clip_index=clip_i)

            avg_series_per_clip[
                clip + '_' + Mode.REST_BETWEEN.value] = rest_vec.tolist()[0]

        return pd.DataFrame.from_dict(avg_series_per_clip)

    @classmethod
    def corr_pipe_single_tr(cls, clip_data, rest_data, tr, re_test=False):

        _tr_mat = cls.compare_clip_to_rest_single_tr(clip_data, rest_data, tr)
        _tr_mat = _tr_mat.apply(lambda x: z_score(x))
        _tr_corr = MatricesOperations.correlation_matrix(_tr_mat)
        _tr_corr = rearrange_clips(_tr_corr, where='rows',
                                   with_testretest=re_test)
        _tr_corr = rearrange_clips(_tr_corr, where='columns',
                                   with_testretest=re_test)
        return _tr_corr



    @classmethod
    def rest_tr_iteration(cls, net, movies, rest, re_test):
        relational_coding = {'correlation': [], 'relation_distance': []}

        for rest_tr in range(0, 19):
            tr_corr = cls.corr_pipe_single_tr(movies, rest, rest_tr, re_test)
            relational_coding['correlation'].append(tr_corr)
            relational_distance = cls.relational_distance(tr_corr.copy())
            relational_coding['relation_distance'].append(relational_distance)

        return relational_coding

    @classmethod
    def executor(cls, with_retest=False):
        relational_coding = {}
        for net in Network:
            rest_data = cls._load_data(Mode.REST_BETWEEN, net)
            clip_data = cls._load_data(Mode.CLIPS, net)
            relational_coding[net.name] = cls.rest_tr_iteration(
                net=net,
                movies=clip_data,
                rest=rest_data,
                re_test=with_retest)


        _dict_to_pkl(relational_coding, "Relational Distance fMRI")


if __name__ == '__main__':
    relational_coding_instance = RelationalCodingfMRI()
    relational_coding_instance.executor()
