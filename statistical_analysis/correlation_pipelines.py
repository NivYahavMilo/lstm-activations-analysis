import itertools
import numpy as np
from statistical_analysis.matrices_ops import MatricesOperations
from statistical_analysis.math_functions import z_score
from mappings.re_arranging import rearrange_clips
import config
import os
import pandas as pd
from enums import Mode
from visualiztions.plot_figure import plot_matrix


def _get_subjects_combinations(subjects_list: list, combine: int = 1):
    # Form all possible correlation combinations
    subjects_list = [*itertools.combinations(subjects_list, combine)]
    return subjects_list


def _load_csv(sub, mode: Mode):
    if len(sub) == 2:
        sub1 = pd.read_csv(os.path.join(config.ACTIVATION_MATRICES, sub[0], mode.value,
                                        'activation_matrix.csv'))
        sub2 = pd.read_csv(os.path.join(config.ACTIVATION_MATRICES, sub[1], mode.value,
                                        'activation_matrix.csv'))
        return sub1, sub2
    elif len(sub) == 1:
        sub = pd.read_csv(os.path.join(config.ACTIVATION_MATRICES, sub[0], mode.value,
                                       'activation_matrix.csv'))
        return sub


def auto_correlation_pipeline(subject: str, mode: Mode):
    """
    function performing auto-correlation for a single subject activation matrix.
    returns a data frame which its columns is flattened matrix represents the correlation for each clip
    """
    corr_per_clip = {}
    sub = _load_csv([subject], mode)
    for clip in list(sub['y'].unique()):
        # Drop all columns unrelated to activation values
        mat = sub[sub['y'] == clip].drop(['y', 'tr'], axis=1)
        # normalize matrix values with z-score
        mat_zscore = mat.apply(lambda x: z_score(x))
        # Calculate Pearson correlation
        pearson_corr = MatricesOperations.auto_correlation_matrix(
            matrix=mat_zscore)
        corr_per_clip[f"{clip}_{mode.value}"] = pearson_corr
    return corr_per_clip


def join_and_auto_correlate(df1: pd.DataFrame, df2: pd.DataFrame):
    clip_vectors = df1.join(df2)
    clips_vectors_z = clip_vectors.apply(lambda x: z_score(x))
    pearson_corr = MatricesOperations.auto_correlation_matrix(
        matrix=clips_vectors_z)
    return pearson_corr


def cross_correlation_pipeline(subject_list):
    corr_per_clip = {}
    subs = _get_subjects_combinations(subject_list)
    # Load a pair
    for sub in subs:
        # Iterate over clips
        sub1, sub2 = _load_csv(sub, mode=Mode.CLIPS)
        for clip in list(sub1['y'].unique()):
            # Drop all columns unrelated to activation values
            mat1 = sub1[sub1['y'] == clip].drop(['y', 'tr'], axis=1)
            mat2 = sub2[sub2['y'] == clip].drop(['y', 'tr'], axis=1)
            # Calculate Pearson correlation
            pearson_corr = MatricesOperations.cross_correlation_matrix(
                matrix1=mat1,
                matrix2=mat2)

            corr_per_clip[clip] = pearson_corr
    return corr_per_clip


def set_activation_vectors(corr: dict):
    clip_activations = pd.DataFrame()
    for clip, corr_mat in corr.items():
        corr_vec: np.array = MatricesOperations.drop_symmetric_side_of_a_matrix(
            matrix=corr_mat,
            drop_diagonal=True)
        clip_activations[clip] = corr_vec
    return clip_activations


def main_pipeline(subjects_list):
    for sub in subjects_list:
        # Execute clip pipeline
        corr_: dict = auto_correlation_pipeline(sub, Mode.CLIPS)
        df_clip: pd.DataFrame = set_activation_vectors(corr_)
        # Execute rest between pipeline
        corr_: dict = auto_correlation_pipeline(sub, Mode.REST_BETWEEN)
        df_rest: pd.DataFrame = set_activation_vectors(corr_)
        # Merging clips and rest between data frame
        corr_mat = join_and_auto_correlate(df_clip, df_rest)
        corr_mat = rearrange_clips(corr_mat, where='rows', with_testretest=False)
        corr_mat = rearrange_clips(corr_mat, where='columns', with_testretest=False)
        corr_mat.to_csv(
            os.path.join(config.ACTIVATION_MATRICES, sub, f'corr_mat.csv'))
        print('done', sub, 'saved to csv')


def create_avg_activation_matrix(table_name: str):
    mat_path = config.ACTIVATION_MATRICES
    all_cor_mat = []
    for sub in os.listdir(mat_path):
        df = pd.read_csv(os.path.join(mat_path, sub, f'{table_name}.csv'), index_col=0)
        matrix: np.array = df.values
        all_cor_mat.append(matrix)

    avg_mat: np.array = MatricesOperations.get_avg_matrix((mat for mat in all_cor_mat))
    avg_mat = pd.DataFrame(avg_mat)
    avg_mat.columns = df.columns
    avg_mat.index = df.index
    avg_mat.to_csv(os.path.join(mat_path, f'average_{table_name}.csv'))
    plot_matrix(avg_mat, title='activations_300roi no test-re-test')


def generate_correlation_per_clip(subject_list, mode: Mode):
    mat_path = config.ACTIVATION_MATRICES
    all_cor_mat = []
    for clip in config.idx_to_clip.values():
        for sub in subject_list:
            corr_mat = pd.read_csv(os.path.join(mat_path, sub, mode.value, 'activation_matrix.csv'))
            matrix = corr_mat[corr_mat['y'] == clip]
            matrix = matrix.drop(['y', 'tr'], axis=1)
            matrix = MatricesOperations.auto_correlation_matrix(matrix)
            all_cor_mat.append(matrix)
        avg_mat: np.array = MatricesOperations.get_avg_matrix(
            (mat for mat in all_cor_mat))
        avg_mat = pd.DataFrame(avg_mat)
        avg_mat.to_csv(os.path.join(
            config.CORRELATION_MATRIX,
            f'average_correlation_matrix_{clip}_{mode.value}.csv'), index=False)


def total_clip_and_rest_correlation(table_name: str):
    df = pd.read_csv(
        os.path.join(
            config.CORRELATION_MATRIX,
            'csvis', f'{table_name}.csv'), index_col=0)

    rest_cor = df.iloc[len(df) // 2:, len(df) // 2:]
    clip_cor = df.iloc[:len(df) // 2, :len(df) // 2]
    df = pd.DataFrame()
    df_rest = MatricesOperations.drop_symmetric_side_of_a_matrix(rest_cor)
    df_clip = MatricesOperations.drop_symmetric_side_of_a_matrix(clip_cor)
    df['clip'] = df_clip
    df['rest'] = df_rest

    df_corr = df.corr()
    return df_corr


if __name__ == '__main__':
    # generate_correlation_per_clip(config.test_list, Mode.CLIPS)
    # generate_correlation_per_clip(config.test_list, Mode.REST_BETWEEN)
    #main_pipeline(config.sub_test_list)
    create_avg_activation_matrix('corr_mat')

    # df = total_clip_and_rest_correlation(table_name='avg_connectivity_300roi.csv')
    print()
