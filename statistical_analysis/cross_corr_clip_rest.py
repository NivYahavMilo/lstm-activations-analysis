import os

import numpy as np
import pandas as pd

import config
from enums import Mode
from statistical_analysis.math_functions import z_score
from statistical_analysis.matrices_ops import MatricesOperations


def avg_activations_per_sub(clip, mode):
    all_cor_mat = []
    for sub in config.sub_test_list:
        corr_mat = pd.read_csv(os.path.join(config.ACTIVATION_MATRICES, sub, mode.value, 'activation_matrix.csv'))
        matrix = corr_mat[corr_mat['y'] == clip]
        matrix = matrix.drop(['y', 'tr'], axis=1)
        all_cor_mat.append(matrix)
    avg_mat: np.array = MatricesOperations.get_avg_matrix(
        (mat for mat in all_cor_mat))
    avg_mat = pd.DataFrame(avg_mat)
    return avg_mat



def clip_rest_cross_corr():
    cross_corr_clip_rest = {}
    for clip in config.idx_to_clip.values():
        movie_clip_avg = avg_activations_per_sub(clip, Mode.CLIPS)
        movie_clip_avg = movie_clip_avg.apply(lambda x: z_score(x))
        rest_between_avg = avg_activations_per_sub(clip, Mode.REST_BETWEEN)
        rest_between_avg = rest_between_avg.apply(lambda x: z_score(x))
        # rest_clip_concat = pd.concat([movie_clip_avg,rest_between_avg])

        cross_corr = MatricesOperations.cross_correlation_matrix(movie_clip_avg, rest_between_avg)
        cross_corr_clip_rest[clip] = cross_corr

    return pd.DataFrame.from_dict(cross_corr_clip_rest)

if __name__ == '__main__':
    df = clip_rest_cross_corr()
    pass