import os
import config
import pandas as pd

from enums import Mode, Network
from statistical_analysis.math_functions import z_score
from statistical_analysis.matrices_ops import MatricesOperations


class ConnectivitySequence:

    @staticmethod
    def get_single_tr_seq(sequence: pd.DataFrame, tr: int):
        if tr == -1:
            tr = int(max(sequence['timepoint'].values))

        seq = sequence[sequence['timepoint'] == tr]
        seq = seq.drop(['y', 'timepoint', 'Subject'], axis=1)
        return seq

    @staticmethod
    def load(mode: Mode, net: Network = None):
        if net:
            data = pd.read_pickle(os.path.join(config.FMRI_DATA_NETWORKS, mode.value,
                                               f"df{net.value}.pkl"))
        else:
            data = pd.read_pickle(os.path.join(config.FMRI_DATA,
                                               f"4_runs_{mode.value}.pkl"))
        return data

    @staticmethod
    def get_avg_clip_seq(data, mode: Mode):
        if mode == Mode.CLIPS:
            return MatricesOperations.get_avg_matrix(iter(data))[0]
        elif mode == Mode.REST_BETWEEN:
            avg_clip_seqs = []
            for tr in range(19):
                avg_tr_seq = MatricesOperations.get_avg_matrix((
                    seq[tr] for seq in data
                ))
                avg_clip_seqs.append(avg_tr_seq[0])
            return avg_clip_seqs

    def get_subjects_average_single_tr(self, clip_i, clip_sequence, net: Network = None):
        df_rest = self.load(Mode.REST_BETWEEN, net)
        all_sub_last_tr_clip = []
        all_sub_rest_trs = []
        for subject in config.sub_test_list:
            sub_data_clip = clip_sequence[clip_sequence['Subject'] == int(subject)]

            sub_data_clip_tr = self.get_single_tr_seq(sub_data_clip, tr=-1)
            all_sub_last_tr_clip.append(sub_data_clip_tr)

            sub_data_rest = df_rest[(df_rest['y'] == clip_i) &
                                    (df_rest['Subject'] == int(subject))]
            sub_data_rest_tr = {}
            for tr in range(19):
                sub_data_rest_tr[tr] = self.get_single_tr_seq(sub_data_rest, tr=tr)
            all_sub_rest_trs.append(sub_data_rest_tr)

        return all_sub_last_tr_clip, all_sub_rest_trs

    def generate_sequence_comparison(self, net, file_name):
        df_clip = self.load(Mode.CLIPS, net)
        correlation_by_tr = {}
        for clip_i in df_clip['y'].unique():
            if clip_i == 0:
                continue
            clip_sequence = df_clip[df_clip['y'] == clip_i]

            subs_clip, subs_rest = self.get_subjects_average_single_tr(clip_i, clip_sequence, net)
            avg_clip_seq = self.get_avg_clip_seq(subs_clip, Mode.CLIPS)
            rest_seqs = self.get_avg_clip_seq(subs_rest, Mode.REST_BETWEEN)
            for tr, seq in enumerate(rest_seqs):
                rest_clip = pd.DataFrame([avg_clip_seq, seq]).T
                rest_clip_zscore = rest_clip.apply(lambda x: z_score(x))
                rest_clip_corr = MatricesOperations.correlation_matrix(
                    rest_clip_zscore)

                correlation_by_tr.setdefault(
                    config.connectivity_mapping[clip_i], []).append(round(
                    rest_clip_corr.loc[0].at[1], 3))

        correlation_by_tr = pd.DataFrame(correlation_by_tr)
        correlation_by_tr.to_csv(
            f"{config.CORRELATION_MATRIX_BY_TR}\\"
            f"{file_name}.csv"
        )
        print(f"Saved {file_name}")
        return correlation_by_tr

    def generate_sequence_comparison_networks(self):
        for net in Network:
            print(f"Working on network {net.name}")
            name = f'fmri {net.value} last tr clip correlation with rest between without test-re-test'
            self.generate_sequence_comparison(net, name)


if __name__ == '__main__':
    corr_seq = ConnectivitySequence()
    corr_seq.generate_sequence_comparison(
        net=None,
        file_name="fmri wb last tr clip correlation with rest between without test-re-test")
    corr_seq.generate_sequence_comparison_networks()
