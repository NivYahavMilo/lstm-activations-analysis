import os

import pandas as pd

ROOT_PATH = os.path.abspath(os.path.curdir)

# Data and outputs live under the repo by default. Override either root with an environment
# variable to point at an external drive without editing this file.
DATA_ROOT = os.environ.get('LSTM_DATA_DIR', ROOT_PATH)
OUTPUT_ROOT = os.environ.get('LSTM_OUTPUT_DIR', ROOT_PATH)

# Outputs: trained models, activation matrices, correlation/connectivity results, figures
MODELS_PATH = os.path.join(OUTPUT_ROOT, 'saved_models')
MODELS_NETWORKS_PATH = os.path.join(OUTPUT_ROOT, 'saved_models', 'networks', 'models')
ACTIVATIONS_NETWORKS_PATH = os.path.join(OUTPUT_ROOT, 'saved_models', 'networks', 'activations')
ACTIVATION_MATRICES = os.path.join(OUTPUT_ROOT, 'activation_matrices')
AVG_ACTIVATION_MATRICES = os.path.join(OUTPUT_ROOT, 'activation_matrices', 'Avg subjects')
CONNECTIVITY_FOLDER = os.path.join(OUTPUT_ROOT, 'fmri_connectivity_matrices')
CORRELATION_MATRIX = os.path.join(OUTPUT_ROOT, 'Correlation Matrix')
CORRELATION_MATRIX_BY_TR = os.path.join(OUTPUT_ROOT, 'Correlation Matrix', 'Correlation by tr')
CORRELATION_MATRIX_REST_CLIP = os.path.join(OUTPUT_ROOT, 'Correlation Matrix',
                                            'Rest-clips correlation')
RESULTS_PATH = os.path.join(OUTPUT_ROOT, 'results')
RESULTS_PATH_NETWORKS = os.path.join(OUTPUT_ROOT, 'results', 'networks')
FIGURES_PATH = os.path.join(OUTPUT_ROOT, 'figures')

# Inputs: preprocessed fMRI data, parcellation files, subject mappings
FMRI_DATA = os.path.join(DATA_ROOT, 'fmri_data')
FMRI_DATA_NETWORKS = os.path.join(DATA_ROOT, 'fmri_data', 'networks_df')
PARCEL_DIR = os.path.join(DATA_ROOT, 'fmri_data', 'cifti')
MAPPINGS_PATH = os.path.join(DATA_ROOT, 'mappings')

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


# The train/test subject split is derived from a data pickle. Compute it lazily on
# first access (PEP 562) so that importing `config` does not require the data file to
# be present — importing the module stays side-effect free.
_subject_split = None


def __getattr__(name):
    global _subject_split
    if name in ('sub_train_list', 'sub_test_list'):
        if _subject_split is None:
            _subject_split = config_data()
        return _subject_split[0] if name == 'sub_train_list' else _subject_split[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
