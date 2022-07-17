import os

import config

import numpy as np
import pandas as pd

from statistical_analysis.matrices_ops import MatricesOperations
from statistical_analysis.math_functions import z_score
from mappings.re_arranging import rearrange_clips
from enums import Mode, Network
from visualiztions.plot_figure import plot_matrix


def _load_csv(sub: str, mode: Mode, net: Network = None):
    if net:
        path = os.path.join(config.ACTIVATION_MATRICES, sub, mode.value, net.value, 'activation_matrix.csv')
    else:
        path = os.path.join(config.ACTIVATION_MATRICES, sub, mode.value, 'activation_matrix.csv')

    sub = pd.read_csv(path)
    return sub


def auto_correlation_pipeline(subject: str, mode: Mode, net: Network = None):
    """
    function performing auto-correlation for a single subject activation matrix.
    returns a data frame which its columns is flattened matrix represents the correlation for each clip
    """
    corr_per_clip = {}
    sub = _load_csv(subject, mode, net)
    for clip in list(sub['y'].unique()):
        # Drop all columns unrelated to activation values
        mat = sub[sub['y'] == clip].drop(['y', 'tr'], axis=1)
        # normalize matrix values with z-score
        mat_zscore = mat.apply(lambda x: z_score(x))
        # Calculate Pearson correlation
        pearson_corr = MatricesOperations.correlation_matrix(
            matrix=mat_zscore)
        corr_per_clip[f"{clip}_{mode.value}"] = pearson_corr
    return corr_per_clip


def join_and_auto_correlate(df1: pd.DataFrame, df2: pd.DataFrame):
    clip_vectors = df1.join(df2)
    clips_vectors_z = clip_vectors.apply(lambda x: z_score(x))
    pearson_corr = MatricesOperations.correlation_matrix(
        matrix=clips_vectors_z)
    return pearson_corr


def set_activation_vectors(corr: dict):
    clip_activations = pd.DataFrame()
    for clip, corr_mat in corr.items():
        corr_vec: np.array = MatricesOperations.drop_symmetric_side_of_a_matrix(
            matrix=corr_mat,
            drop_diagonal=True)
        clip_activations[clip] = corr_vec
    return clip_activations


def generate_correlation_per_clip(subject_list, mode: Mode):
    mat_path = config.ACTIVATION_MATRICES
    all_cor_mat = []
    for clip in config.idx_to_clip.values():
        for sub in subject_list:
            corr_mat = pd.read_csv(os.path.join(mat_path, sub, mode.value, 'activation_matrix.csv'))
            matrix = corr_mat[corr_mat['y'] == clip]
            matrix = matrix.drop(['y', 'tr'], axis=1)
            matrix = MatricesOperations.correlation_matrix(matrix)
            all_cor_mat.append(matrix)
        avg_mat: np.array = MatricesOperations.get_avg_matrix(
            (mat for mat in all_cor_mat))
        avg_mat = pd.DataFrame(avg_mat)
        avg_mat.to_csv(os.path.join(
            config.CORRELATION_MATRIX,
            f'avg_corr_mat{clip}_{mode.value}.csv'), index=False)


def total_clip_and_rest_correlation(table_name: str):
    df = pd.read_csv(f'{table_name}.csv', index_col=0)

    rest_cor = df.iloc[len(df) // 2:, len(df) // 2:]
    clip_cor = df.iloc[:len(df) // 2, :len(df) // 2]
    df = pd.DataFrame()
    df_rest = MatricesOperations.drop_symmetric_side_of_a_matrix(rest_cor)
    df_clip = MatricesOperations.drop_symmetric_side_of_a_matrix(clip_cor)
    df['clip'] = df_clip
    df['rest'] = df_rest

    df_corr = df.corr()
    return df_corr


def main_pipeline(subjects_list, table_name, net: Network = None, re_test: bool = False):
    for sub in subjects_list:
        # Execute clip pipeline
        corr_: dict = auto_correlation_pipeline(sub, Mode.CLIPS, net)
        df_clip: pd.DataFrame = set_activation_vectors(corr_)
        # Execute rest between pipeline
        corr_: dict = auto_correlation_pipeline(sub, Mode.REST_BETWEEN, net)
        df_rest: pd.DataFrame = set_activation_vectors(corr_)
        # Merging clips and rest between data frame
        corr_mat = join_and_auto_correlate(df_clip, df_rest)
        corr_mat = rearrange_clips(corr_mat, where='rows', with_testretest=re_test)
        corr_mat = rearrange_clips(corr_mat, where='columns', with_testretest=re_test)
        corr_mat.to_csv(os.path.join(config.ACTIVATION_MATRICES, sub, f'{table_name}.csv'))
        print('done', sub, 'saved to csv', net.name)


def create_avg_activation_matrix(table_name: str):
    mat_path = config.ACTIVATION_MATRICES
    all_cor_mat = []
    subjects = os.listdir(mat_path)
    for sub in subjects:
        if sub.endswith('csv'):
            continue
        df = pd.read_csv(os.path.join(mat_path, sub, f'{table_name}.csv'), index_col=0)
        matrix: np.array = df.values
        all_cor_mat.append(matrix)

    avg_mat: np.array = MatricesOperations.get_avg_matrix((mat for mat in all_cor_mat))
    avg_mat = pd.DataFrame(avg_mat)
    avg_mat.columns = df.columns
    avg_mat.index = df.index
    avg_mat.to_csv(os.path.join(mat_path, f'avg {table_name}.csv'))
    plot_matrix(avg_mat, title=table_name)


def wb_pipeline():
    # generate_correlation_per_clip(config.test_list, Mode.CLIPS)
    # generate_correlation_per_clip(config.test_list, Mode.REST_BETWEEN)

    table = 'corr mat wb without test-re-test'
    main_pipeline(config.sub_test_list, table, re_test=False)
    create_avg_activation_matrix(table)

    clip_rest_corr = total_clip_and_rest_correlation(f"{config.ACTIVATION_MATRICES}\\avg {table}")
    clip_rest_corr.to_csv('rest-clip corr' + table + '.csv')

    table = 'corr mat wb with test-re-test'
    main_pipeline(config.sub_test_list, table, re_test=True)
    create_avg_activation_matrix(table)

    clip_rest_corr = total_clip_and_rest_correlation(f"{config.ACTIVATION_MATRICES}\\avg {table}")
    clip_rest_corr.to_csv('rest-clip corr' + table + '.csv')


def net_pipeline():
    for net in Network:
        table = f'{net.value} corr mat without test-re-test'
        main_pipeline(config.sub_test_list, table, net, re_test=False)
        create_avg_activation_matrix(table)

        clip_rest_corr = total_clip_and_rest_correlation(f"{config.ACTIVATION_MATRICES}\\avg {table}")
        clip_rest_corr.to_csv('rest-clip corr' + table + '.csv')


if __name__ == '__main__':
    net_pipeline()
