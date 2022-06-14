"""
script for generating connectivity matrix out of 300 roi from fmri data
"""
import pandas as pd

import config
from enums import Mode
from mappings.re_arranging import rearrange_clips
from statistical_analysis.correlation_pipelines import set_activation_vectors, join_and_auto_correlate
from statistical_analysis.math_functions import z_score
from statistical_analysis.matrices_ops import MatricesOperations
from visualiztions.plot_figure import plot_matrix

class RoiConnectivity:
    def __init__(self):
        self.data = {Mode.CLIPS: self.__load_fmri_data('movie_runs'),
                     Mode.REST_BETWEEN: self.__load_fmri_data('runs_rest_between')}
        self.test_subjects = config.sub_test_list.astype(int)
        self.clips = config.connectivity_mapping.keys()

    @staticmethod
    def __load_fmri_data(filename):
        return pd.read_pickle(f"{config.FMRI_DATA}//4_{filename}.pkl")

    def auto_corr(self, sub, mode:Mode):
        corr_per_clip = {}
        for clip in list(sub['y'].unique()):
            if clip==0:
                continue
            # Drop all columns unrelated to activation values
            mat = sub[sub['y'] == clip].drop(['y', 'timepoint', 'Subject'], axis=1)
            # normalize matrix values with z-score
            mat_zscore = mat.apply(lambda x: z_score(x))
            # Calculate Pearson correlation
            pearson_corr = MatricesOperations.auto_correlation_matrix(
                matrix=mat_zscore)
            clip_name = config.connectivity_mapping.get(clip)
            corr_per_clip[f"{clip_name}_{mode.value}"] = pearson_corr
        return corr_per_clip

    def pipe(self, table_name, re_test: bool = False):
        corr_mat = []
        for sub_ in self.test_subjects:
            sub_clip = self.data[Mode.CLIPS].copy()
            sub = sub_clip[sub_clip['Subject']==sub_]
            corr_: dict = self.auto_corr(sub, Mode.CLIPS)
            df_clip: pd.DataFrame = set_activation_vectors(corr_)
            # Execute rest between pipeline
            sub_rest = self.data[Mode.REST_BETWEEN].copy()
            sub = sub_rest[sub_rest['Subject']==sub_]
            corr_: dict = self.auto_corr(sub, Mode.REST_BETWEEN)
            df_rest: pd.DataFrame = set_activation_vectors(corr_)
            # Merging clips and rest between data frame
            corr_mat.append(join_and_auto_correlate(df_clip, df_rest))
            print(f"Done subject {sub_}")

        print("averaging all connectivity matrices....")

        headers = corr_mat[0].columns
        avg_conn_mat = pd.DataFrame(MatricesOperations.get_avg_matrix(iter(corr_mat)))
        avg_conn_mat.columns = headers
        avg_conn_mat.index = headers

        rearrange_clips(avg_conn_mat, where='columns', with_testretest=re_test)
        rearrange_clips(avg_conn_mat, where='rows', with_testretest=re_test)
        avg_conn_mat.to_csv(f"{table_name}.csv")
        plot_matrix(pd.DataFrame(avg_conn_mat), title=table_name)



if __name__ == '__main__':
    roi = RoiConnectivity()
    table = "avg connectivity wb without test-re-test"
    roi.pipe(table, re_test=False)

    roi = RoiConnectivity()
    table = "avg connectivity wb with test-re-test"
    roi.pipe(table, re_test=True)
