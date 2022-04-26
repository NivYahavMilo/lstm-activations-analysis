import os
import pandas as pd
import numpy as np

import config
from statistical_analysis.correlation_pipelines import set_activation_vectors, auto_correlation_pipeline, \
    join_and_auto_correlate, create_avg_activation_matrix
from statistical_analysis.math_functions import z_score
from statistical_analysis.matrices_ops import MatricesOperations
from statistical_analysis.table_builder import Mode
from supporting_functions import _load_csv


def get_custom_tr(mat_clip):
    # getting interest indices
    stop = int(max(mat_clip['tr'].values))
    start = stop - 19
    start_s: int = mat_clip[mat_clip['tr'] == start].index.values[0]
    stop_s: int = mat_clip[mat_clip['tr'] == stop].index.values[0]
    mat_clip_prune: pd.DataFrame = mat_clip.filter(items = [*range(start_s, stop_s)], axis=0)
    return mat_clip_prune


def auto_correlation_pipeline_custom_tr(subject: str, mode: Mode):
    corr_per_clip = {}
    mat_path = os.path.join(config.ACTIVATION_MATRICES, subject, mode.value, 'activation_matrix.csv')
    sub = _load_csv(mat_path)
    for clip in list(sub['y'].unique()):
        # Drop all columns unrelated to activation values
        mat = sub[sub['y'] == clip]
        mat_pruned = get_custom_tr(mat)
        mat_pruned = mat_pruned.drop(['y', 'tr'], axis=1)
        # normalize matrix values with z-score
        mat_zscore = mat_pruned.apply(lambda x: z_score(x))
        # Calculate Pearson correlation
        pearson_corr = MatricesOperations.auto_correlation_matrix(
            matrix=mat_zscore)
        corr_per_clip[f"{clip}_{mode.value}"] = pearson_corr
    return corr_per_clip

def main_correlation_tr_pipeline(sub_list):
    for sub in sub_list:
        # Execute clip pipeline
        corr_dict = auto_correlation_pipeline_custom_tr(sub, Mode.CLIPS)
        df_clip: pd.DataFrame = set_activation_vectors(corr_dict)
        # Execute rest between pipeline
        corr_: dict = auto_correlation_pipeline(sub, Mode.REST_BETWEEN)
        df_rest: pd.DataFrame = set_activation_vectors(corr_)
        # Merging clips and rest between data frame
        corr_mat = join_and_auto_correlate(df_clip, df_rest)
        corr_mat.to_csv(
            os.path.join(config.ACTIVATION_MATRICES, sub, f'correlation_matrix_19_tr.csv'))
        print('done', sub, 'saved to csv')


if __name__ == '__main__':

    #main_correlation_tr_pipeline(config.sub_test_list)

    create_avg_activation_matrix('correlation_matrix_19_tr')