import pandas as pd
import os
import glob
from pathlib import Path
from utils import create_dir_if_nonexist
from features import clean_text_label


print('Creating source label vocab')

# Switches
remove_punctuation = True
split_labels_by_whitespace = True

# Constants
empty_col_name = 'Unnamed: 0'
n_root = 6357

# use glob to get all the csv files in the folder
work_dir = os.environ['work_dir'] + '/'
conc_files = glob.glob(os.path.join(work_dir + 'training_concs_hscpc/', "*.xlsx"))

dictionary = []

# loop over the list of csv files
for f in conc_files:

    fname = Path(f).name

    print('.')
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

print('.')

# Delete duplicates
len_prior = len(dictionary)
dictionary = list(dict.fromkeys(dictionary))
len_new = len(dictionary)

print('Removed ' + str(len_prior-len_new) + ' duplicate entries; ' + str(len_new) + ' entries remain')

# Save
save_path = work_dir + '/training_data/'
create_dir_if_nonexist(save_path)

fname = save_path + 'label_dictionary.pkl'
label_store = pd.DataFrame(dictionary, columns=['source_labels'])
label_store.to_pickle(fname)

print('Finished')