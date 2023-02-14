import pandas as pd
import numpy as np
from utils import duplicates_in_list
from sklearn.cluster import KMeans
from matplotlib import pyplot
import random


def combine_estimation_methods():

    return None


def conc_flood_fill_com(conc_estimated, threshold=0.95):
    """
        Flood fills from center of mass
    """

    conc_estimated_tr = conc_estimated.transpose().reset_index()
    conc_estimated_tr['index_int'] = conc_estimated_tr.index
    original_cols = conc_estimated.index.to_list()

    # Find center of mass for each source category
    positions = []
    for c in original_cols:

        # Test whether the threshold is less than the column minimum
        if conc_estimated_tr[c].max()*0.95 < threshold:
            tmp_threshold = conc_estimated_tr[c].max()*0.95
        else:
            tmp_threshold = threshold

        # Select values larger than the threshold
        c_top_prob = conc_estimated_tr.loc[conc_estimated_tr[c] > tmp_threshold][[c, 'index_int']]
        count_matches = c_top_prob.shape[0]

        assert count_matches > 0, 'No predictions for source label: ' + c

        probs = c_top_prob[c].values.reshape(-1, 1)
        idxs = c_top_prob['index_int'].values.reshape(-1, 1)
        com = np.average(idxs, axis=0, weights=probs**3)
        positions.append(int(com))

    # Initialise at the com points
    conc_ar = np.zeros(conc_estimated.shape, dtype=int)
    conc_est_ar = conc_estimated.to_numpy()
    conc_rel_imp = conc_est_ar / conc_est_ar.sum(axis=1).reshape(-1, 1)  # relative importances

    for i, j in enumerate(positions):
        if sum(conc_ar[:, j]) == 0:
            conc_ar[i, j] = 1
        elif sum(conc_ar[:, j+1]) == 0:
            conc_ar[i, j+1] = 1
        elif sum(conc_ar[:, j - 1]) == 0:
            conc_ar[i, j - 1] = 1
    assert all(np.sum(conc_ar, axis=0) <= 1), 'Source matches are not unique!'

    # Flood fill
    k = 1
    hard_max_iter = 15000
    max_iter = 4000
    n_columns = conc_estimated.shape[1]

    while sum(sum(conc_ar)) < n_columns and k <= hard_max_iter:

        for i, row in enumerate(conc_ar):

            # Indices already set in this row
            matches = np.nonzero(row)
            assert matches is not None

            if matches[0] is not None and len(matches[0]) > 0:

                # Choose direction in which to move
                move = [min(matches[0]) - 1, max(matches[0]) + 1]
                possible_moves = [x for x in move if 0 <= x <= n_columns-1]  # not reached edge

                if possible_moves is not None and possible_moves != []:
                    next_idx = random.choice(possible_moves)
                    if sum(conc_ar[:, next_idx]) == 0:  # column has no entries in other rows
                        prob_new = conc_est_ar[i, next_idx]
                        if prob_new >= min(min(conc_est_ar[i, matches])):
                            conc_ar[i, next_idx] = 1
                        elif prob_new * (1+(k/max_iter)) >= np.quantile(conc_est_ar[i, :], 0.98):
                            conc_ar[i, next_idx] = 1
                        elif conc_rel_imp[i, next_idx] * (1+(0.8*k/max_iter)) > max(conc_rel_imp[:, next_idx]):
                            conc_ar[i, next_idx] = 1
                        elif np.sum(conc_ar[i, 0:next_idx], axis=0) == 0:  # lower edge
                            assert sum(conc_ar[:, next_idx]) == 0
                            conc_ar[i, next_idx] = 1
                        elif np.sum(conc_ar[i, next_idx:], axis=0) == 0:  # upper edge
                            assert sum(conc_ar[:, next_idx]) == 0
                            conc_ar[i, next_idx] = 1

        k = k + 1

        # Log progress
        if k % 100 == 0:
            completeness = sum(conc_ar.reshape(-1))/n_columns * 100
            print('k: ' + str(k) + ', fill: ' + str(round(completeness)) + '% complete')

    # Tests
    assert all(np.sum(conc_ar, axis=0) <= 1), 'Source matches are not unique!'

    # Convert to dataframe
    conc_com = pd.DataFrame(conc_ar, columns=conc_estimated.columns, index=conc_estimated.index, dtype=int)

    return conc_com


def maximum_match_probability(conc_estimated):

    conc_max_prob = pd.DataFrame(0, columns=conc_estimated.columns, index=conc_estimated.index, dtype=int)
    conc_estimated_tr = conc_estimated.transpose()

    # Highest match for each source label
    for row_index, row in conc_max_prob.iterrows():

        # HSCPC probabilties in top quantile
        top_ranked_matches = conc_estimated_tr[row_index].quantile(0.998)

        filter_matches = conc_estimated_tr[conc_estimated_tr[row_index] >= top_ranked_matches][row_index]
        assert not filter_matches.empty

        # Column indices
        col_idxs_hscpc = filter_matches.index.to_list()
        assert not duplicates_in_list(col_idxs_hscpc)

        # Set value
        conc_max_prob.loc[row_index, col_idxs_hscpc] = 1

    # Check columns for unallocated
    for c in conc_max_prob.columns:
        if conc_max_prob[c].sum() == 0:
            idx_max = conc_estimated[c].idxmax()
            if not isinstance(idx_max, str):
                idx_max = idx_max[0]
            conc_max_prob.at[idx_max, c] = 1

    return conc_max_prob


def conc_from_decision_boundary(conc_estimated, decision_boundary=0.95):

    conc_estimated[conc_estimated >= decision_boundary] = 1
    conc_estimated[conc_estimated < decision_boundary] = 0

    return conc_estimated


def conc_decision_boundary_voting_ensemble(conc_raw):
    """
        Averages across a range of potential decision boundaries
    """

    d_bs = [0.9, 0.95, 0.98, 0.99, 0.9999]
    weights = [0.1, 0.2, 0.3, 0.3, 0.1]

    conc_sheets = np.zeros((conc_raw.shape[0], conc_raw.shape[1], len(d_bs)))

    for i, d in enumerate(d_bs):

        conc_d = conc_raw.to_numpy()
        conc_d[conc_d >= d] = 1
        conc_d[conc_d < d] = 0
        conc_sheets[:, :, i] = conc_d

    conc_all = np.average(conc_sheets, axis=2, weights=weights)
    conc_final = pd.DataFrame(conc_all, columns=conc_raw.columns, index=conc_raw.index)

    return conc_final


def conc_flood_fill_max_prob(conc_estimated):
    """
        Flood fills from the highest probability in each row
    """

    conc_est_tr = conc_estimated.transpose().reset_index()
    conc_est_tr['index_int'] = conc_est_tr.index
    original_cols = conc_estimated.index.to_list()

    conc_ar = np.zeros(conc_estimated.shape, dtype=int)

    # Find highest probability each source category
    for i, c in enumerate(original_cols):

        probs_sorted = np.sort(conc_est_tr[c].values)[::-1]
        filled = False
        k = 0
        previous_prob = 1

        while not filled and k <= 10:

            prob = probs_sorted[k]
            c_top_prob = conc_est_tr.loc[(conc_est_tr[c] >= prob) & (conc_est_tr[c] < previous_prob)][[c, 'index_int']]

            if c_top_prob.shape[0] == 0:
                if prob == previous_prob:
                    c_top_prob = conc_est_tr.loc[(conc_est_tr[c] >= prob)]

            j = c_top_prob['index_int'].values[0]
            if sum(conc_ar[:, j]) == 0:
                conc_ar[i, j] = 1
                filled = True

            k = k + 1
            previous_prob = prob

    # Flood fill
    conc_est_ar = conc_estimated.to_numpy()
    conc_rel_imp = conc_est_ar / conc_est_ar.sum(axis=1).reshape(-1, 1)  # relative importances

    k = 1
    hard_max_iter = 15000
    max_iter = 4000
    n_columns = conc_estimated.shape[1]

    while sum(sum(conc_ar)) < n_columns and k <= hard_max_iter:

        for i, row in enumerate(conc_ar):

            # Indices already set in this row
            matches = np.nonzero(row)

            # Choose direction in which to move
            move = [min(matches[0]) - 1, max(matches[0]) + 1]
            possible_moves = [x for x in move if 0 <= x <= n_columns-1]  # not reached edge

            if possible_moves is not None and possible_moves != []:
                next_idx = random.choice(possible_moves)
                if sum(conc_ar[:, next_idx]) == 0:  # column has no entries in other rows
                    prob_new = conc_est_ar[i, next_idx]
                    if prob_new >= min(min(conc_est_ar[i, matches])):
                        conc_ar[i, next_idx] = 1
                    elif prob_new * (1+(k/max_iter)) >= np.quantile(conc_est_ar[i, :], 0.98):
                        conc_ar[i, next_idx] = 1
                    elif conc_rel_imp[i, next_idx] * (1+(0.8*k/max_iter)) > max(conc_rel_imp[:, next_idx]):
                        conc_ar[i, next_idx] = 1
                    elif np.sum(conc_ar[i, 0:next_idx], axis=0) == 0:  # lower edge
                        assert sum(conc_ar[:, next_idx]) == 0
                        conc_ar[i, next_idx] = 1
                    elif np.sum(conc_ar[i, next_idx:], axis=0) == 0:  # upper edge
                        assert sum(conc_ar[:, next_idx]) == 0
                        conc_ar[i, next_idx] = 1

        k = k + 1

        # Log progress
        if k % 100 == 0:
            completeness = sum(conc_ar.reshape(-1))/n_columns * 100
            print('k: ' + str(k) + ', fill: ' + str(round(completeness)) + '% complete')

    # Tests
    assert all(np.sum(conc_ar, axis=0) <= 1), 'Source matches are not unique!'

    # Convert to dataframe
    conc_com = pd.DataFrame(conc_ar, columns=conc_estimated.columns, index=conc_estimated.index, dtype=int)

    return conc_com


def conc_flood_fill_max_prob_root_category(conc_in):
    """
        Flood fills from the highest probability in each root category
    """

    root_cols = conc_in.columns.to_list()
    source_cols = conc_in.index.to_list()
    conc_in = conc_in.reset_index(drop=True)

    conc_ar = np.zeros(conc_in.shape, dtype=int)

    conc_in['index_int'] = conc_in.index

    # Find highest probability in each root category
    for j, c in enumerate(root_cols):

        i_top_prob = conc_in.loc[conc_in[c] == conc_in[c].max()].head(1)['index_int'].values[0]
        if sum(conc_ar[i_top_prob, :]) == 0:
            conc_ar[i_top_prob, j] = 1

    # Transpose
    conc_in_tr = conc_in.drop(labels=['index_int'], axis=1).transpose()
    conc_in_tr.columns = source_cols
    conc_in_tr = conc_in_tr.reset_index(drop=True)
    conc_ar = conc_ar.transpose()

    # Find highest probability each source category
    for j, c in enumerate(source_cols):

        # Source category currently has no matches
        if sum(conc_ar[:, j]) == 0:

            probs_sorted = np.sort(conc_in_tr[c].values)[::-1]
            filled = False
            k = 0
            previous_prob = 1

            while not filled and k <= 10:

                prob = probs_sorted[k]
                c_top_prob = conc_in_tr.loc[(conc_in_tr[c] >= prob) & (conc_in_tr[c] < previous_prob)]

                if c_top_prob.shape[0] == 0 and prob == previous_prob:
                    c_top_prob = conc_in_tr.loc[(conc_in_tr[c] >= prob)]

                i = c_top_prob.index.values[0]
                if sum(conc_ar[i, :]) == 0:
                    conc_ar[i, j] = 1
                    filled = True

                k = k + 1
                previous_prob = prob

    # Flood fill
    conc_ar = conc_ar.transpose()
    conc_est_ar = conc_in.drop(labels=['index_int'], axis=1).to_numpy()
    conc_rel_imp = conc_est_ar / conc_est_ar.sum(axis=1).reshape(-1, 1)  # relative importances

    k = 1
    hard_max_iter = 15000
    max_iter = 4000
    n_columns = conc_ar.shape[1]

    while sum(sum(conc_ar)) < n_columns and k <= hard_max_iter:

        for i, row in enumerate(conc_ar):

            # Indices already set in this row
            matches = np.nonzero(row)

            # Choose direction in which to move
            move = [min(matches[0]) - 1, max(matches[0]) + 1]
            possible_moves = [x for x in move if 0 <= x <= n_columns-1]  # not reached edge

            if possible_moves is not None and possible_moves != []:
                next_idx = random.choice(possible_moves)
                if sum(conc_ar[:, next_idx]) == 0:  # column has no entries in other rows
                    prob_new = conc_est_ar[i, next_idx]
                    if prob_new >= min(min(conc_est_ar[i, matches])):
                        conc_ar[i, next_idx] = 1
                    elif prob_new * (1+(k/max_iter)) >= np.quantile(conc_est_ar[i, :], 0.98):
                        conc_ar[i, next_idx] = 1
                    elif conc_rel_imp[i, next_idx] * (1+(0.8*k/max_iter)) > max(conc_rel_imp[:, next_idx]):
                        conc_ar[i, next_idx] = 1
                    elif np.sum(conc_ar[i, 0:next_idx], axis=0) == 0:  # lower edge
                        assert sum(conc_ar[:, next_idx]) == 0
                        conc_ar[i, next_idx] = 1
                    elif np.sum(conc_ar[i, next_idx:], axis=0) == 0:  # upper edge
                        assert sum(conc_ar[:, next_idx]) == 0
                        conc_ar[i, next_idx] = 1

        k = k + 1

        # Log progress
        if k % 100 == 0:
            completeness = sum(conc_ar.reshape(-1))/n_columns * 100
            print('k: ' + str(k) + ', fill: ' + str(round(completeness)) + '% complete')

    # Tests
    assert all(np.sum(conc_ar, axis=0) <= 1), 'Source matches are not unique!'

    # Convert to dataframe
    conc_com = pd.DataFrame(conc_ar, columns=root_cols, index=source_cols, dtype=int)

    return conc_com