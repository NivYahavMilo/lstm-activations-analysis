import os
import pandas as pd

ROOT_PATH = os.path.abspath(os.path.curdir)
MODELS_PATH = os.path.join(ROOT_PATH, 'saved_models')
MODELS_NETWORKS_PATH = os.path.join(ROOT_PATH, 'saved_models', 'networks', 'models')
ACTIVATIONS_NETWORKS_PATH = os.path.join(ROOT_PATH, 'saved_models', 'networks', 'activations')
ACTIVATION_MATRICES = os.path.join(ROOT_PATH, 'activation_matrices')
AVG_ACTIVATION_MATRICES = os.path.join(ROOT_PATH, 'activation_matrices', 'Avg subjects')
FMRI_DATA = os.path.join(ROOT_PATH, 'fmri_data')
FMRI_DATA_NETWORKS = os.path.join(ROOT_PATH, 'fmri_data', 'networks_df')
CONNECTIVITY_FOLDER = os.path.join(ROOT_PATH, 'fmri_connectivity_matrices')
MAPPINGS_PATH = os.path.join(ROOT_PATH, 'mappings')
CORRELATION_MATRIX = os.path.join(ROOT_PATH, 'Correlation Matrix')
CORRELATION_MATRIX_BY_TR = os.path.join(ROOT_PATH, 'Correlation Matrix', 'Correlation by tr')
CORRELATION_MATRIX_REST_CLIP = os.path.join(ROOT_PATH, 'Correlation Matrix', 'Rest-clips correlation')
RESULTS_PATH = os.path.join(ROOT_PATH, 'results')
RESULTS_PATH_NETWORKS = os.path.join(ROOT_PATH, 'results', 'networks')

TEST_SUBJECTS_AMOUNT = 76
TRAIN_SUBJECTS_AMOUNT = 100

idx_to_clip = {0: 'testretest1',
               1: 'testretest2',
               2: 'testretest3',
               3: 'testretest4',
               4: 'twomen',
               5: 'bridgeville',
               6: 'pockets',
               7: 'overcome',
               8: 'inception',
               9: 'socialnet',
               10: 'oceans',
               11: 'flower',
               12: 'hotel',
               13: 'garden',
               14: 'dreary',
               15: 'homealone',
               16: 'brokovich',
               17: 'starwars',
               }

connectivity_mapping = {0: 'testretest',
                        1: 'twomen',
                        2: 'bridgeville',
                        3: 'pockets',
                        4: 'overcome',
                        5: 'inception',
                        6: 'socialnet',
                        7: 'oceans',
                        8: 'flower',
                        9: 'hotel',
                        10: 'garden',
                        11: 'dreary',
                        12: 'homealone',
                        13: 'brokovich',
                        14: 'starwars',
                        }


def config_data():
    df = pd.read_pickle(os.path.join(FMRI_DATA, '4_runs_rest_between.pkl'))
    subject_list = df['Subject'].astype(str).unique()
    sub_train_list_ = subject_list[:TRAIN_SUBJECTS_AMOUNT]
    sub_test_list_ = subject_list[TRAIN_SUBJECTS_AMOUNT:]
    return sub_train_list_, sub_test_list_


sub_train_list, sub_test_list = config_data()
