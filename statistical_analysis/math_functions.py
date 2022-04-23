import numpy as np
import pandas as pd


def z_score(seq: np.array):
    seq = (1 / np.std(seq)) * (seq - np.mean(seq))
    return seq

def pearson_correlation(seq: pd.DataFrame):
    corr = (1/len(seq)-1) * (seq.T * seq)
    return corr

