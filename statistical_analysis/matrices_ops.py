import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List


class MatricesOperations:

    @classmethod
    def generate_heat_map(cls, corr_series: pd.Series):
        plt.matshow(corr_series)
        plt.show()

    @classmethod
    def cross_correlation_matrix(cls, matrix1: pd.DataFrame, matrix2: pd.DataFrame,
                                     method: str = 'pearson'):
        pearson_corr = matrix1.corrwith(matrix2, axis=0, method=method)
        return pearson_corr

    @classmethod
    def correlation_matrix(cls, matrix: pd.DataFrame):
        return matrix.corr()

    @classmethod
    def flatten_matrix(cls, matrix: np.array):
        return matrix.flatten()

    @classmethod
    def is_symmetric(cls, matrix, rtol=1e-05, atol=1e-08):
        return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)

    @classmethod
    def drop_symmetric_side_of_a_matrix(cls, matrix: pd.DataFrame, drop_diagonal: bool = True):
        if not cls.is_symmetric(matrix):
            raise ValueError("Input matrix shape should be symmetric")
        # Copy so we never mutate the caller's matrix; the previous implementation wrote
        # NaNs into the input's diagonal in place.
        values = np.asarray(matrix).copy()
        h = values.shape[0]
        # k = -1 excludes the main diagonal; k = 0 keeps it.
        k = -1 if drop_diagonal else 0
        lower_triangle: np.array = values[np.tril_indices(h, k=k)]
        return lower_triangle

    @classmethod
    def plot_matrix_as_image(cls, matrix:pd.DataFrame):
        pass

    @classmethod
    def get_avg_matrix(cls, matrices: iter, axis: int = 0):
        matrices: List[np.array] = [mat for mat in matrices]
        avg_mat: np.array = np.mean(matrices, axis=axis)
        return avg_mat

    @classmethod
    def get_avg_vector(cls, vec: pd.Series):
        return vec.mean()

    @classmethod
    def clip_rest_correlation(cls, corr_matrix: pd.DataFrame):
        """Correlate the clip vs. rest halves of a stacked correlation matrix.

        The top-left quadrant is the clip-vs-clip correlation block and the bottom-right
        quadrant is the rest-vs-rest block. Each is flattened (symmetric side dropped) into a
        vector; the returned 2x2 frame is the correlation between the ``clip`` and ``rest``
        vectors.
        """
        half = len(corr_matrix) // 2
        clip_cor = corr_matrix.iloc[:half, :half]
        rest_cor = corr_matrix.iloc[half:, half:]
        flat = pd.DataFrame()
        flat['clip'] = cls.drop_symmetric_side_of_a_matrix(clip_cor)
        flat['rest'] = cls.drop_symmetric_side_of_a_matrix(rest_cor)
        return flat.corr()



