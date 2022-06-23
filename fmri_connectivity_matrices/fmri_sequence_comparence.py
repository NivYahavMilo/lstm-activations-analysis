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
            tr = int(max(sequence['tr'].values))

        seq = sequence[sequence['tr'] == tr]
        seq = seq.drop(['y', 'tr'], axis=1)
        return seq

    @staticmethod
    def load(mode: Mode):
        data = pd.read_pickle(os.path.join(
            config.FMRI_DATA,
            f"4_{mode.value}_runs.pkl"))

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

    def get_subjects_average_single_tr(self, clip_sequence, net: Network = None):
        all_sub_last_tr_clip = []
        all_sub_rest_trs = []
        for sub in clip_sequence:
            sub_data_clip = clip_sequence[clip_sequence['Subject'] == sub]

            sub_data_clip_tr = self.get_single_tr_seq(sub_data_clip, tr=-1)
            all_sub_last_tr_clip.append(sub_data_clip_tr)
        #     #
        #     # sub_data = self.load(data, Mode.REST_BETWEEN)
        #     # sub_data_rest = sub_data[sub_data['y'] == clip]
        #     sub_data_rest_tr = {}
        #     for tr in range(19):
        #         sub_data_rest_tr[tr] = self.get_single_tr_seq(sub_data_rest, tr=tr)
        #     all_sub_rest_trs.append(sub_data_rest_tr)
        #
        # return all_sub_last_tr_clip, all_sub_rest_trs

    def iterate_clips(self):
        data = self.load(Mode.CLIPS)
        for clip in data['y'].unique():
            clip_sequence = data[data['y']==clip]
            subs_clip, subs_rest = self.get_subjects_average_single_tr(clip)
            clip_seq = self.get_avg_clip_seq(subs_clip, Mode.CLIPS)
            rest_seqs = self.get_avg_clip_seq(subs_rest, Mode.REST_BETWEEN)
            for tr, seq in enumerate(rest_seqs):
                rest_clip = pd.DataFrame([clip_seq, seq]).T
                rest_clip_zscore = rest_clip.apply(lambda x: z_score(x))
                rest_clip_corr = MatricesOperations.auto_correlation_matrix(
                    rest_clip_zscore)

                correlation_by_tr.setdefault(clip, []).append(round(rest_clip_corr.loc[0].at[1], 3))

        pd.DataFrame(correlation_by_tr).to_csv(
            f"{config.CORRELATION_MATRIX_BY_TR}\\last tr clip correlation with rest between.csv"
        )

if __name__ == '__main__':
    corr_seq = ConnectivitySequence()
    corr_seq.iterate_clips()
