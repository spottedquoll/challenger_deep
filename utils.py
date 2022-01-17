import numpy as np
import pandas as pd
import os
from collections.abc import Sequence
import pickle


def duplicates_in_list(lst):

    return len(lst) != len(set(lst))


def write_pickle(fname, obj_to_write):

    with open(fname, 'wb') as f:
        pickle.dump(obj_to_write, f)


def read_pickle(fname):

    with open(fname, 'rb') as pickle_file:
        result = pickle.load(pickle_file)

    return result


def get_shape(lst, shape=()):
    """
    returns the shape of nested lists similarly to numpy's shape.

    :param lst: the nested list
    :param shape: the shape up to the current recursion depth
    :return: the shape including the current depth
    https://stackoverflow.com/questions/51960857/how-can-i-get-a-list-shape-without-using-numpy/51961584
    """

    if not isinstance(lst, Sequence):
        # base case
        return shape

    # peek ahead and assure all lists in the next depth
    # have the same length
    if isinstance(lst[0], Sequence):
        l = len(lst[0])
        if not all(len(item) == l for item in lst):
            msg = 'not all lists have the same length'
            raise ValueError(msg)

    shape += (len(lst),)

    # recurse
    shape = get_shape(lst[0], shape)

    return shape


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