import os
from utils import read_pickle
import numpy as np
import pandas as pd
from feature_engineering import encode_source_labels, clean_text_label, create_position_feature
from utils import duplicates_in_list


# Switches
model_version = 'model_2022-01-15_w2882_s3714'
feature_meta_version = 'feature_meta_2022-01-16_w2882'
target_label_file = 'USA_BEA_403_labels'
decision_boundary = 0.8

print('Predicting ' + target_label_file + ' HSCPC matches')

# Constants
n_root = 6357

# Paths
work_dir = os.environ['work_dir'] + '/'
model_dir = work_dir + 'model/'
prediction_dir = work_dir + 'predictions/'

# Extract target labels
df = pd.read_excel(prediction_dir + target_label_file + '.xlsx', header=None)
source_labels = df[0].values.tolist()
x_labels = []

for i, r in enumerate(source_labels):
    row_label = clean_text_label(r)
    x_labels.append(row_label)

print('Extracted ' + str(len(x_labels)) + ' source labels')

assert not duplicates_in_list(x_labels)

# Load HSCPC labels
df = pd.read_excel(work_dir + 'hscpc_labels.xlsx', header=0)
target_labels = df['Labels'].values

assert not duplicates_in_list(x_labels)

# Load feature meta
feature_meta_fname = work_dir + 'model/' + feature_meta_version + '.pkl'
feature_meta = read_pickle(feature_meta_fname)

tokenizer = feature_meta['tokenizer']
max_words = feature_meta['max_words']

# One-hot-encode x labels
x_features_encoded = encode_source_labels(tokenizer, x_labels, max_words)

# Add position features
if feature_meta['add_position_features']:
    position_feature = create_position_feature(len(x_labels), n_root)
    x_features_encoded = np.append(x_features_encoded, np.array(position_feature), axis=1)

# Load predictive model
model_fname = model_dir + model_version + '.pkl'
model = read_pickle(model_fname)

# Make prediction
source_name = target_label_file.replace('_labels', '')
preds = model.predict(x_features_encoded)

# Save raw estimates
conc_estimated = pd.DataFrame(preds, index=x_labels, columns=target_labels)
conc_estimated.to_excel(prediction_dir + 'challenger_deep_predict_' + source_name + '_raw' + '.xlsx')

# Save estimates, filtered by decision boundary
est = preds.copy()
est[est >= decision_boundary] = 1
est[est < decision_boundary] = 0

conc_filtered = pd.DataFrame(est, index=x_labels, columns=target_labels, dtype=int)
conc_filtered.to_excel(prediction_dir + 'challenger_deep_predict_' + source_name + '_conc_decision_'
                       + str(decision_boundary) + '.xlsx')

# Set binary values based on th maximum probability in each row and column
conc_max_prob = pd.DataFrame(0, columns=conc_estimated.columns, index=conc_estimated.index, dtype=int)

# Maximum probability in each column
for c in conc_estimated.columns:
    idx_max = conc_estimated[c].idxmax()
    if not isinstance(idx_max, str):
        idx_max = idx_max[0]
    conc_max_prob.at[idx_max, c] = 1

# If first pass did left rows unset, set column link using max row probability
row_max_idxs = conc_estimated.idxmax(axis="columns")
for row_index, row in conc_max_prob.iterrows():
    if row.sum() == 0:
        col_index = row_max_idxs[row_index]
        conc_max_prob.at[row_index, col_index] = 1

conc_max_prob.to_excel(prediction_dir + 'challenger_deep_predict_' + source_name + '_conc_max_probability'
                       + '.xlsx')

