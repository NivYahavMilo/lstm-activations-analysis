import os

from typing import Dict, List

import numpy as np

import config
import enums
import pandas as pd

from statistical_analysis.math_functions import z_score
from statistical_analysis.matrices_ops import MatricesOperations


class Connectivity:

    def __init__(self, mode: enums.Mode, net: enums.Network):
        self.mode = mode
        self.net = net
        self.avg_con_per_clip: Dict[List] = {}
        self.net_df: pd.DataFrame = self._load_net()
        self.subjects = list(self.net_df['Subject'].unique())
        self.clips = list(self.net_df['y'].unique())

    def _load_net(self):
        net = pd.read_pickle(os.path.join(config.FMRI_DATA_NETWORKS,
                                          self.mode.value,
                                          ''.join(['df', self.net.value, '.pkl'])))
        return net

    def _split_into_cases(self):
        for sub in self.subjects:
            for clip in self.clips:
                yield sub, self.net_df[(self.net_df['y'] == clip) &
                                       (self.net_df['Subject'] == sub)]

    def _correlate_fmri_data(self):
        for sub, case in self._split_into_cases():
            case = case.drop(['Subject', 'y', 'timepoint'], axis=1)
            # normalize matrix values with z-score
            case_zscore = case.apply(lambda x: z_score(x))
            # correlation per single clip for a single subject
            corr_net = MatricesOperations.auto_correlation_matrix(case_zscore)
            self.avg_con_per_clip.setdefault(sub, []).append(corr_net)

    def _average_all_cases(self):
        for clip in self.clips:
            all_cor_mat = []
            for sub in self.subjects:
                all_cor_mat.append(self.avg_con_per_clip[sub][clip])

            avg_mat: np.array = MatricesOperations.get_avg_matrix(
                (mat for mat in all_cor_mat))
            clip_name = config.connectivity_mapping[clip]
            pd.DataFrame(avg_mat).to_csv(os.path.join(
                config.CONNECTIVITY_FOLDER, self.net.value, self.mode.value, f"connectivity_matrix_{clip_name}.csv"))
            print(f"Saved {clip_name} clip for {self.net.name} in mode {self.mode.name} to csv")

    @classmethod
    def generate_connectivity_matrices(cls):
        for mode in enums.Mode:
            for net in enums.Network:
                path = os.path.join(config.CONNECTIVITY_FOLDER, net.value, mode.value)
                if not os.path.exists(path):
                    os.makedirs(path)
                connectivity = cls(mode=mode, net=net)
                connectivity._correlate_fmri_data()
                connectivity._average_all_cases()
                print(f"Done {net.name} in mode {mode.name}")


if __name__ == '__main__':
    Connectivity.generate_connectivity_matrices()
