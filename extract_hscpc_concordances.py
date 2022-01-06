import pandas as pd
import os
import glob
from pathlib import Path

# Constants
n_root = 6357
empty_col_name = 'Unnamed: 0'

# use glob to get all the csv files in the folder
work_dir = os.environ['work_dir'] + '/'
conc_files = glob.glob(os.path.join(work_dir + 'training_concs_hscpc/', "*.xlsx"))

# loop over the list of csv files
for f in conc_files:

    fname = Path(f).name
    print('Reading ' + f)

    # Read the concordance file
    df = pd.read_excel(f)

    # Check the dimensions
    columns = df.columns
    assert len(columns) == n_root + 1

    # Get the source labels
    assert columns[0] == empty_col_name
    source_labels = df[[empty_col_name]].values
    df.rename(columns={empty_col_name: "labels"}, inplace=True)

    # Set labels as row indexes
    conc = df.set_index('labels')

    # Convert to array
    conc_ar = conc.to_numpy(dtype=int, copy=True)  # what do NaNs get converted as?

    assert conc_ar.shape[0] == len(source_labels)
    assert conc_ar.shape[1] == n_root

