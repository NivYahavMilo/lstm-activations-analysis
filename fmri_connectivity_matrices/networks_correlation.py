import os

from enums import Network,Mode
import pandas as pd
import config
from statistical_analysis.math_functions import z_score
from statistical_analysis.matrices_ops import MatricesOperations


def load_connectivity_matrix(mode, net):
    path = os.path.join(config.CONNECTIVITY_FOLDER, net.value, mode.value)
    for movie in os.listdir(path):
        data = pd.read_csv(os.path.join(path, movie), index_col=0)
        movie_name = f"{mode.value}{net.value}{movie[movie.rfind('_'):movie.find('.')]}"
        yield movie_name, data

def pull_triangle_from_corr_matrix(mode:Mode, net: Network):
    data_mapping = {}
    # Matrix shape ROIxROI
    for movie, mat in load_connectivity_matrix(mode, net):
        # Vector shaped (ROI*(ROI-1)/2)x1
        flatt = MatricesOperations.drop_symmetric_side_of_a_matrix(mat)
        data_mapping[movie] = flatt
    # flattened correlation vectors for each movie
    data_frame = pd.DataFrame.from_dict(data_mapping)
    # Matrix shaped (MOVIES x FLATT)
    data_frame = data_frame.apply(lambda x: z_score(x))
    # apply correlation between the fmri movie vectors
    # Matrix shaped (MOVIE x MOVIE)
    movie_corr = MatricesOperations.correlation_matrix(data_frame)
    # flatten results
    # Vector shaped (MOVIE*(MOVIE-1)/2)x1
    movie_corr_flat = MatricesOperations.drop_symmetric_side_of_a_matrix(movie_corr)
    return movie_corr_flat

def correlate_flattened_representation():
    movies_vectors = {}
    for mode in Mode:
        for net in Network:
            net_vec = pull_triangle_from_corr_matrix(mode, net)
            movies_vectors[net.name] = net_vec
        data_frame = pd.DataFrame.from_dict(movies_vectors)
        cor_mat_all_nets = MatricesOperations.correlation_matrix(data_frame)


if __name__ == '__main__':
    correlate_flattened_representation()