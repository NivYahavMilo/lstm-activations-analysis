import pandas as pd
from abc import abstractmethod

from enums import Network, Mode
from statistical_analysis.matrices_ops import MatricesOperations


class RelationalCoding:

    @staticmethod
    @abstractmethod
    def _load_data(mode: Mode, network: Network):
        pass

    @staticmethod
    def get_single_tr(mat_clip: pd.DataFrame, clip_i: int, tr_field: str, tr_pos: int = -1):
        if tr_pos == -1:
            xdf = mat_clip[mat_clip['y'] == clip_i]
            tr_pos = int(max(xdf[tr_field].values))

        sequence = mat_clip[(mat_clip[tr_field] == tr_pos) &
                            (mat_clip['y'] == clip_i)]

        sequence = sequence.drop(['Subject', 'y', tr_field], axis=1)
        return sequence

    @classmethod
    def relational_distance(cls, df):
        rest_cor = df.iloc[len(df) // 2:, len(df) // 2:]
        clip_cor = df.iloc[:len(df) // 2, :len(df) // 2]
        df = pd.DataFrame()
        df_rest = MatricesOperations.drop_symmetric_side_of_a_matrix(rest_cor)
        df_clip = MatricesOperations.drop_symmetric_side_of_a_matrix(clip_cor)
        df['clip'] = df_clip
        df['rest'] = df_rest

        df_corr = df.corr()
        distance = round(df_corr.loc['clip'].at['rest'], 3)
        return distance