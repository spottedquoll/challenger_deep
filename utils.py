import numpy as np
import pandas as pd
import os


def is_empty(obj):

    if obj is None:
        return True
    elif isinstance(obj, np.float):
        return False
    elif isinstance(obj, pd.DataFrame):
        if obj.empty:
            return True
        else:
            return False
    elif isinstance(obj, list):
        if not obj:  # obj == []
            return True
        else:
            return False
    elif len(obj) == 0:
        return True
    elif len(obj) > 0:
        return False
    else:
        raise ValueError('Could not determine non/existence')


def create_dir_if_nonexist(path):
    if not os.path.exists(path):
        os.makedirs(path)