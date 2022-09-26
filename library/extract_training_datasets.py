import pandas as pd
import os
import glob
from pathlib import Path
import numpy as np
from utils import is_empty
from library.feature_engineering import clean_text_label, dictionary_supplement


def extract_concordances_into_rows(training_data_path, save_path, n_root=6357):

    print('Extracting training data from HSCPC concordances')

    # Constants
    empty_col_name = 'Unnamed: 0'

    # Discover all the training concs in the folder
    conc_files = glob.glob(os.path.join(training_data_path, "*.xlsx"))

    feature_store = []

    # loop over the list of csv files
    for f in conc_files:

        fname = Path(f).name

        print('.')
        print('Extracting ' + fname)

        if '.xlsx' in fname and '~$' not in fname:

            # Read the concordance file
            df = pd.read_excel(f, sheet_name=0, header=0)

            # Check the dimensions
            columns = df.columns
            assert len(columns) == n_root + 1

            # Get the source labels
            assert columns[0] == empty_col_name
            source_labels = df[[empty_col_name]].values
            df.rename(columns={empty_col_name: "labels"}, inplace=True)

            n_source = len(source_labels)

            # Set labels as row indexes
            conc = df.set_index('labels')

            # Convert to array
            assert not conc.isnull().values.any()
            conc_ar = conc.to_numpy(dtype=int, copy=True)  # what do NaNs get converted as?

            assert conc_ar.shape[0] == n_source and conc_ar.shape[1] == n_root

            tmp_store = []
            for i, r in enumerate(conc_ar):

                row_label = str(source_labels[i][0])
                row_label = clean_text_label(row_label)

                assert np.min(r) >= 0 and np.max(r) <= 1, ('Non-binary values found on line: ' + str(i) +
                                                           ', label: ' + row_label)

                # Find indices of HSCPC labels
                if np.sum(r) > 0:

                    # Indices of non-zero elements
                    nnzs = np.where(r > 0)[0]
                    assert not is_empty(nnzs)
                    hscpc_labels = nnzs

                    # Position feature
                    position = (i+1)/conc_ar.shape[0]

                    # Store
                    tmp_store.append({'conc': fname, 'source_row_label': row_label, 'hscpc_labels': hscpc_labels, 'position': position})

            feature_store.extend(tmp_store)

            print('Added ' + str(len(tmp_store)) + ' rows')

    print('Feature store contains: ' + str(len(feature_store)) + ' rows in total')

    fname = save_path + 'hscpc_collected_training_set.pkl'
    feature_fd = pd.DataFrame(feature_store)
    feature_fd.to_pickle(fname)

    print('Finished extracting concordances')


def get_training_data(raw_data_dir, training_data_dir, extract_training_data=True):

    if extract_training_data:
        extract_concordances_into_rows(raw_data_dir, training_data_dir)

    training_data = pd.read_pickle(training_data_dir + 'hscpc_collected_training_set.pkl')

    return training_data


def create_source_label_vocabularly(raw_data_dir, training_data_dir, n_root):

    print('Creating source label vocab')

    # Switches
    remove_punctuation = True
    split_labels_by_whitespace = True

    # Constants
    empty_col_name = 'Unnamed: 0'

    # use glob to get all the csv files in the folder
    conc_files = glob.glob(os.path.join(raw_data_dir, "*.xlsx"))
    assert conc_files is not None and len(conc_files) > 0

    dictionary = []

    # loop over the list of csv files
    for f in conc_files:

        fname = Path(f).name

        print('Extracting ' + fname)

        tmp_store = []

        if '.xlsx' in fname and '~$' not in fname:

            # Read the concordance file
            df = pd.read_excel(f)

            # Check the dimensions
            columns = df.columns
            assert len(columns) == n_root + 1

            # Get the source labels
            assert columns[0] == empty_col_name
            source_labels = df[[empty_col_name]].values

            for s in source_labels:

                # Extract
                lbl = s[0]

                #  clean the label
                if remove_punctuation:
                    lbl = clean_text_label(lbl)

                # Split
                if split_labels_by_whitespace:
                    lbl = lbl.split(' ')

                # Append
                tmp_store.extend(lbl)

            dictionary.extend(tmp_store)

    print('Extracted dictionary length: ' + str(len(dictionary)))

    # Supplement dictionary
    dictionary.extend(dictionary_supplement())

    print('Post supplemented length: ' + str(len(dictionary)))

    # Delete duplicates
    len_prior = len(dictionary)
    dictionary = list(dict.fromkeys(dictionary))
    len_new = len(dictionary)

    print('Removed ' + str(len_prior - len_new) + ' duplicate entries; ' + str(len_new) + ' entries remain')

    # Delete unuseful words
    words = ['and', 'or']

    # Save
    fname = training_data_dir + 'label_dictionary.pkl'
    label_store = pd.DataFrame(dictionary, columns=['source_labels'])
    label_store.to_pickle(fname)

    print('Finished building vocabularly')


def extract_train_concs_as_sequence(training_data_path, save_path, n_root=6357):

    print('Extracting training data from HSCPC concordances')

    # Constants
    empty_col_name = 'Unnamed: 0'

    # Discover all the training concs in the folder
    conc_files = glob.glob(os.path.join(training_data_path, "*.xlsx"))

    feature_store = []

    # loop over the list of csv files
    for f in conc_files:

        fname = Path(f).name

        print('... ' + fname)

        if '.xlsx' in fname and '~$' not in fname:

            # Read the concordance file
            df = pd.read_excel(f, sheet_name=0, header=0)

            # Check the dimensions
            columns = df.columns
            assert len(columns) == n_root + 1

            # Get the source labels
            assert columns[0] == empty_col_name
            source_labels = df[[empty_col_name]].values
            df.rename(columns={empty_col_name: "labels"}, inplace=True)

            n_source = len(source_labels)

            # Set labels as row indexes
            conc = df.set_index('labels')

            # Convert to array
            assert not conc.isnull().values.any()
            conc_ar = conc.to_numpy(dtype=int, copy=True)  # what do NaNs get converted as?

            assert conc_ar.shape[0] == n_source and conc_ar.shape[1] == n_root

            tmp_store = []
            hscpc_seq = ''
            src_lbl_seq = ''

            for i, r in enumerate(conc_ar):

                row_label = str(source_labels[i][0])
                row_label = clean_text_label(row_label)

                assert all(np.isfinite(r))
                assert np.min(r) >= 0 and np.max(r) <= 1, ('Non-binary values found on line: ' + str(i) +
                                                           ', label: ' + row_label)

                # Find indices of HSCPC labels
                if np.sum(r) > 0:

                    # Indices of non-zero elements
                    nnzs = np.where(r > 0)[0]
                    assert not is_empty(nnzs)
                    hscpc_labels = nnzs

                    # Make sequences
                    hscpc_seq = hscpc_seq + ' ZZZZ ' + ' '.join([str(x) for x in hscpc_labels])
                    src_lbl_seq = src_lbl_seq + ' ZZZZ ' + row_label

            # Store
            tmp_store.append({'conc': fname, 'source_label_sequence': src_lbl_seq, 'hscpc_sequence': hscpc_seq})
            feature_store.extend(tmp_store)

    print('Feature store contains: ' + str(len(feature_store)))

    fname = save_path + 'sequence_training_set.pkl'
    feature_fd = pd.DataFrame(feature_store)
    feature_fd.to_pickle(fname)

    print('Finished extracting concordances')