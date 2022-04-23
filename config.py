import os
import pandas as pd

ROOT_PATH = os.path.abspath(os.path.curdir)
MODELS_PATH = os.path.join(ROOT_PATH, 'save_models')
DATA_PATH = os.path.join(ROOT_PATH, 'data')
ACTIVATION_MATRICES = os.path.join(DATA_PATH, 'activations_matrices')

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

train_size = 100
df = pd.read_pickle(os.path.join(DATA_PATH, '4_runs_rest_between.pkl'))
subject_list = df['Subject'].astype(str).unique()
clip_names = list(df['y'].unique())
train_list = subject_list[:train_size]
test_list = subject_list[train_size:]

del df,subject_list
