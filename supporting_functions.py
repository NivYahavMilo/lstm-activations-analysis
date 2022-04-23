import os
import pickle
import config

def _dict_to_pkl(data: dict, file_name: str):

    with open(os.path.join(config.MODELS_PATH, f'{file_name}.pkl'), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pkl(file_name: str):
    file = open(os.path.join(config.MODELS_PATH, file_name), 'rb')
    object_file = pickle.load(file)
    return object_file
