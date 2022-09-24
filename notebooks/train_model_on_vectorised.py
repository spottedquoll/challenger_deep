import os
from library.training_toolkit import extract_concordances_single_row
from utils import create_dir_if_nonexist

# Switches
extract_training_data = True

# Paths
work_dir = os.environ['work_dir']
raw_data_dir = work_dir + 'training_concs_hscpc/'
training_data_dir = work_dir + 'training_data/'
create_dir_if_nonexist(training_data_dir)
model_dir = work_dir + 'model/'
create_dir_if_nonexist(model_dir)

# Extract training data
if extract_training_data:
    extract_concordances_single_row(raw_data_dir, training_data_dir)