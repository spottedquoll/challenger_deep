import os
from utils import read_pickle
import numpy as np
import pandas as pd
from library.feature_engineering import encode_x_labels, clean_text_label, create_position_feature, make_c100_features
from utils import duplicates_in_list
from library.make_estimated_conc import (maximum_match_probability, conc_flood_fill_com, conc_from_decision_boundary,
                                         conc_flood_fill_max_prob)

# Switches
model_version = 'model_2022-09-24_w3126_s4521_l4'
feature_meta_version = 'feature_meta_2022-09-24_w3126'
target_label_file = 'USA_BEA_15_labels'
decision_boundary = 0.85

print('Predicting ' + target_label_file + ' HSCPC matches')

# Constants
n_root = 6357

# Paths
work_dir = os.environ['work_dir'] + '/'
model_dir = work_dir + 'model/'
prediction_dir = work_dir + 'predictions/'
label_dir = work_dir + 'input_labels/'

# Extract target labels
df = pd.read_excel(label_dir + target_label_file + '.xlsx', header=None)
source_labels = df[0].values.tolist()
x_labels = []

for i, r in enumerate(source_labels):
    row_label = clean_text_label(r)
    x_labels.append(row_label)

print('Extracted ' + str(len(x_labels)) + ' source labels')

assert not duplicates_in_list(x_labels)

# Load HSCPC labels
df = pd.read_excel(work_dir + 'hscpc/hscpc_labels.xlsx', header=0)
target_labels = df['Labels'].values

assert not duplicates_in_list(x_labels)

# Load feature meta
feature_meta_fname = work_dir + 'model/' + feature_meta_version + '.pkl'
feature_meta = read_pickle(feature_meta_fname)

# Unpack tokenizer
tokenizer = feature_meta['tokenizer']
sequences = feature_meta['sequences']
max_words = feature_meta['max_words']
x_feature_one_hot_encoding = feature_meta['x_feature_one_hot_encoding']

# One-hot-encode x labels
x_features_encoded = encode_x_labels(sequences, x_labels, max_words, one_hot_encoding=x_feature_one_hot_encoding)

# Add label position feature
if feature_meta['add_position_features']:
    position_feature = create_position_feature(x_labels)
    x_features_encoded = np.hstack((x_features_encoded, np.array(position_feature).reshape(-1, 1)))

# C100 features
if feature_meta['add_isic_100_features']:
    c100_labels = pd.read_excel(work_dir + 'hscpc/c100_labels.xlsx')['sector'].to_list()
    new_features = make_c100_features(x_labels, c100_labels)
    x_features_encoded = np.hstack((x_features_encoded, new_features))

# Load predictive model
model_fname = model_dir + model_version + '.pkl'
model = read_pickle(model_fname)

# Make prediction
preds = model.predict(x_features_encoded)

# Save predictions
print('Saving concordance predictions')

source_name = target_label_file.replace('_labels', '')
fname_prefix = prediction_dir + 'challenger_deep_predict_' + source_name

# Save raw estimates
conc_raw_estimates = pd.DataFrame(preds, index=x_labels, columns=target_labels)
conc_raw_estimates.to_excel(fname_prefix + '_raw' + '.xlsx')

# Save estimates, filtered by decision boundary
conc_filtered = conc_from_decision_boundary(conc_raw_estimates.copy(), decision_boundary=decision_boundary)
conc_filtered.to_excel(fname_prefix + '_conc_thresholded_' + str(decision_boundary) + '.xlsx')

# Center of mass
conc_com = conc_flood_fill_com(conc_raw_estimates.copy())
conc_com.to_excel(fname_prefix + '_conc_flood_fill_com' + '.xlsx')

# Set binary values based on the maximum probability in each row
conc_max_prob = maximum_match_probability(conc_raw_estimates.copy())
conc_max_prob.to_excel(fname_prefix + '_conc_max_prob' + '.xlsx')

# Center of mass
conc_com = conc_flood_fill_max_prob(conc_raw_estimates.copy())
conc_com.to_excel(fname_prefix + '_conc_flood_fill_max_prob' + '.xlsx')

print('Finished writing predictons to ' + prediction_dir)