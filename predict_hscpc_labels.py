import os
from utils import read_pickle
import numpy as np
import pandas as pd
from feature_engineering import encode_source_labels, clean_text_label, create_position_feature
from utils import duplicates_in_list
from make_estimated_conc import maximum_match_probability, conc_from_clusters, conc_flood_fill_com, conc_from_decision_boundary

# Switches
model_version = 'model_2022-02-25_w3075_s4318'  # 'model_2022-01-15_w2882_s3714'
feature_meta_version = 'feature_meta_2022-02-25_w3075'  # 'feature_meta_2022-01-16_w2882'
target_label_file = 'USA_BEA_15_labels'
decision_boundary = 0.8

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

print('Saving concordance predictions')
fname_prefix = prediction_dir + 'challenger_deep_predict_' + source_name

# Save raw estimates
conc_raw_estimates = pd.DataFrame(preds, index=x_labels, columns=target_labels)
conc_raw_estimates.to_excel(fname_prefix + '_raw' + '.xlsx')

# Save estimates, filtered by decision boundary
conc_filtered = conc_from_decision_boundary(conc_raw_estimates.copy(), decision_boundary=0.7)
conc_filtered.to_excel(fname_prefix + '_conc_thresholded_' + str(decision_boundary) + '.xlsx')

# Center of mass
conc_com = conc_flood_fill_com(conc_raw_estimates)
conc_com.to_excel(fname_prefix + '_conc_flood_fill_com' + '.xlsx')

# Set binary values based on the maximum probability in each row
conc_max_prob = maximum_match_probability(conc_raw_estimates.copy())
conc_max_prob.to_excel(fname_prefix + '_conc_max_prob' + '.xlsx')


print('Finished')