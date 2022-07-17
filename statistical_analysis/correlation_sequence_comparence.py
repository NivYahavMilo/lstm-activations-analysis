import os
import config
import pandas as pd
import matplotlib.pyplot as plt

from enums import Mode, Network
from statistical_analysis.math_functions import z_score
from statistical_analysis.matrices_ops import MatricesOperations


class CorrelationSequence:

    @staticmethod
    def get_single_tr_seq(sequence: pd.DataFrame, tr: int):
        if tr == -1:
            tr = int(max(sequence['tr'].values))
        seq = sequence[sequence['tr'] == tr]
        seq = seq.drop(['y', 'tr'], axis=1)
        return seq

    @staticmethod
    def load(table_path):
        data = pd.read_csv(f"{table_path}//activation_matrix.csv")
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

    def get_subjects_average_single_tr(self, clip, net: Network = None):
        all_sub_last_tr_clip = []
        all_sub_rest_trs = []
        for sub in os.listdir(config.ACTIVATION_MATRICES):
            if not sub.isdigit():
                continue
            sub_path = os.path.join(config.ACTIVATION_MATRICES, sub, Mode.CLIPS.value)
            if net:
                sub_path = os.path.join(sub_path, net.value)
            sub_data = self.load(sub_path)
            sub_data_clip = sub_data[sub_data['y'] == clip]
            sub_data_clip_tr = self.get_single_tr_seq(sub_data_clip, tr=-1)
            all_sub_last_tr_clip.append(sub_data_clip_tr)
            sub_path = os.path.join(config.ACTIVATION_MATRICES, sub, Mode.REST_BETWEEN.value)
            if net:
                sub_path = os.path.join(sub_path, net.value)
            sub_data = self.load(sub_path)
            sub_data_rest = sub_data[sub_data['y'] == clip]
            sub_data_rest_tr = {}
            for tr in range(19):
                sub_data_rest_tr[tr] = self.get_single_tr_seq(sub_data_rest, tr=tr)
            all_sub_rest_trs.append(sub_data_rest_tr)

        return all_sub_last_tr_clip, all_sub_rest_trs

    def generate_sequence_comparison(self, net: Network, table_name):
        correlation_by_tr = {}
        for clip in config.idx_to_clip.values():
            if clip.startswith('test'):
                continue
            subs_clip, subs_rest = self.get_subjects_average_single_tr(clip, net)
            clip_seq = self.get_avg_clip_seq(subs_clip, Mode.CLIPS)
            rest_seqs = self.get_avg_clip_seq(subs_rest, Mode.REST_BETWEEN)
            for tr, seq in enumerate(rest_seqs):
                rest_clip = pd.DataFrame([clip_seq, seq]).T
                rest_clip_zscore = rest_clip.apply(lambda x: z_score(x))
                rest_clip_corr = MatricesOperations.correlation_matrix(
                    rest_clip_zscore)

                correlation_by_tr.setdefault(clip, []).append(round(rest_clip_corr.loc[0].at[1], 3))

        correlation_by_tr = pd.DataFrame(correlation_by_tr)
        correlation_by_tr.to_csv(
            f"{config.CORRELATION_MATRIX_BY_TR}\\"
            f"{table_name}.csv"
        )
        print(f"Saved {table_name}")
        return correlation_by_tr

    def generate_sequence_comparison_networks(self):
        for net in Network:
            print(f"Working on network {net.name}")
            name = f'lstm patterns {net.value} last tr clip correlation with rest between without test-re-test'
            self.generate_sequence_comparison(net, name)


if __name__ == '__main__':
    corr_seq = CorrelationSequence()

    corr_seq.generate_sequence_comparison(
        net=None,
        table_name="lstm patterns WB last tr clip correlation with rest between without test-re-test.csv")
    corr_seq.generate_sequence_comparison_networks()
