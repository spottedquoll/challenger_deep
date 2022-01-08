import pandas as pd
import os
import glob
from pathlib import Path
import numpy as np
from utils import is_empty, create_dir_if_nonexist

print('Extracting training data from HSCPC concordances')

# Constants
n_root = 6357
empty_col_name = 'Unnamed: 0'

# use glob to get all the csv files in the folder
work_dir = os.environ['work_dir'] + '/'
conc_files = glob.glob(os.path.join(work_dir + 'training_concs_hscpc/', "*.xlsx"))

feature_store = []

# loop over the list of csv files
for f in conc_files:

    fname = Path(f).name

    print('.')
    print('Extracting ' + fname)

    if '.xlsx' in fname and '~$' not in fname: 

        # Read the concordance file
        df = pd.read_excel(f)

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
        conc_ar = conc.to_numpy(dtype=int, copy=True)  # what do NaNs get converted as?

        assert conc_ar.shape[0] == n_source
        assert conc_ar.shape[1] == n_root

        tmp_store = []
        for i, r in enumerate(conc_ar):

            row_label = str(source_labels[i][0])
            assert np.min(r) >= 0 and np.max(r) <= 1, 'Non-binary values found on line: ' + str(i) + ', label: ' + row_label

            if np.sum(r) > 0:

                # Find HSCPC labels
                nnzs = np.where(r > 0)[0]
                assert not is_empty(nnzs)

                # Store
                hscpc_labels = ','.join(nnzs.astype(str))
                tmp_store.append({'conc': fname, 'source_row_label': row_label, 'hscpc_labels': hscpc_labels,
                                  'position': (i+1)/n_source, 'row_weight': np.sum(r)/np.sum(conc_ar)})

        feature_store.extend(tmp_store)

        print('Added ' + str(len(tmp_store)) + ' rows')

print('Feature store contains: ' + str(len(feature_store)) + ' rows in total')

# Save
save_path = work_dir + '/training_data/'
create_dir_if_nonexist(save_path)

fname = save_path + 'hscpc_collected_training_set.xlsx'
pd.DataFrame(feature_store).to_excel(fname)

print('Finished')