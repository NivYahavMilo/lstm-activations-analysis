import numpy as np
import pandas as pd


def z_score(seq: np.array):
    seq = (1 / np.std(seq)) * (seq - np.mean(seq))
    return seq

def pearson_correlation(seq: pd.DataFrame):
    """Pearson correlation matrix between the columns of ``seq``.

    The previous implementation was ``(1/len(seq)-1) * (seq.T * seq)`` which, besides the
    operator-precedence slip (``1/len - 1`` instead of ``1/(len-1)``), used an element-wise
    product rather than a matrix product and never normalized by the standard deviations, so
    it did not compute a correlation at all.
    """
    centered = seq - seq.mean()
    cov = centered.T.dot(centered) / (len(seq) - 1)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return corr

