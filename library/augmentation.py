import random

import numpy as np


def augment_by_union(x, y, position_feature, factor):
    """
        Performs augmentation by joining records, the text labels and corresponding x indices of two different records are joined
    """

    n_real_records = len(x)
    extra_rows = int(n_real_records*factor)

    for i in range(extra_rows):

        rand_idx = random.randint(0, n_real_records-2)
        new_x_label = x[rand_idx] + ' ' + x[rand_idx+1]
        new_y_label = np.unique(np.concatenate((y[rand_idx], y[rand_idx+1])))
        new_position = np.mean([position_feature[rand_idx][2], position_feature[rand_idx+1][2]])

        x.append(new_x_label)
        y.append(new_y_label)
        position_feature.append([position_feature[rand_idx][0], position_feature[rand_idx][1], new_position])

    print('Training x augmented from ' + str(n_real_records) + ' to ' + str(len(x)) + ' records')

    return x, y, position_feature


def augment_by_adjacent_union(x, y, position_feature, factor):
    """
        Performs augmentation by joining records
        The text labels and corresponding x indices of two adjacent records are joined
    """

    n_real_records = len(x)
    extra_rows = int(n_real_records*factor)

    for i in range(extra_rows):

        # Select 2 indices to join
        rand_idx = random.randint(0, n_real_records-2)

        # Find a partner where at least one index overlaps
        partner_idx = None
        k = 0
        while k <= 25 and partner_idx is None:
            candidate_idx = random.randint(0, n_real_records - 2)
            if candidate_idx != rand_idx and len(set(y[rand_idx]).intersection(set(y[candidate_idx]))) > 0:
                partner_idx = candidate_idx
            k = k + 1

        # Create joined record
        if partner_idx is not None:

            new_x_label = x[rand_idx] + ' ' + x[rand_idx+1]
            new_y_label = np.unique(np.concatenate((y[rand_idx], y[partner_idx])))
            new_position = np.mean([position_feature[rand_idx][2], position_feature[partner_idx][2]])

            # Augment new record to training data
            x.append(new_x_label)
            y.append(new_y_label)
            position_feature.append([position_feature[rand_idx][0], position_feature[rand_idx][1], new_position])

    print('Training x augmented from ' + str(n_real_records) + ' to ' + str(len(x)) + ' records')

    return x, y, position_feature