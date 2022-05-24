import os
import pandas as pd

ROOT_PATH = os.path.abspath(os.path.curdir)
MODELS_PATH = os.path.join(ROOT_PATH, 'saved_models')
ACTIVATION_MATRICES = os.path.join(ROOT_PATH, 'activation_matrices')
FMRI_DATA = os.path.join(ROOT_PATH, 'fmri_data')
FMRI_DATA_NETWORKS = os.path.join(ROOT_PATH, 'fmri_data', 'networks_df')
CONNECTIVITY_FOLDER = os.path.join(ROOT_PATH, 'fmri_connectivity_matrices')
MAPPINGS_PATH = os.path.join(ROOT_PATH, 'mappings')

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

train_size = 100


def config_data():
    df = pd.read_pickle(os.path.join(FMRI_DATA, '4_runs_rest_between.pkl'))
    subject_list = df['Subject'].astype(str).unique()
    clip_names_ = list(df['y'].unique())
    sub_train_list_ = subject_list[:train_size]
    sub_test_list_ = subject_list[train_size:]
    return clip_names_, sub_train_list_, sub_test_list_


clip_names, sub_train_list, sub_test_list = config_data()
