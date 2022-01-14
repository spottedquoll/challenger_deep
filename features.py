import string
import random
import numpy as np


def augment_by_union(x, y, position_feature, factor):

    n_real_records = len(x)
    extra_rows = int(n_real_records*factor)

    for i in range(extra_rows):

        rand_idx = random.randint(0, n_real_records-2)
        new_x_label = x[rand_idx] + ' ' + x[rand_idx+1]
        new_y_label = np.unique(np.concatenate((y[rand_idx], y[rand_idx+1])))
        new_position = np.mean([position_feature[rand_idx], position_feature[rand_idx+1]])

        x.append(new_x_label)
        y.append(new_y_label)
        position_feature = np.concatenate((position_feature, np.expand_dims(new_position, axis=-1)))

    print('Training data augmented from ' + str(n_real_records) + ' to ' + str(len(x)) + ' records')

    return x, y, position_feature


def clean_text_label(txt):

    # Remove punctuation and make lower case
    lbl = txt.lower()
    for c in string.punctuation:
        lbl = lbl.replace(c, " ")

    # Remove numeric characters
    lbl = ''.join([i for i in lbl if not i.isdigit()])

    # Remove double whitespace
    lbl = lbl.replace('   ', ' ')
    lbl = lbl.replace('  ', ' ')

    return lbl

