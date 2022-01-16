import os
from utils import read_pickle
import numpy as np
import pandas as pd
from feature_engineering import encode_source_labels, clean_text_label, create_position_feature

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

# Load HSCPC labels
df = pd.read_excel(work_dir + 'hscpc_labels.xlsx', header=0)
target_labels = df['Labels'].values

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

estimated_concordance = pd.DataFrame(preds, index=x_labels, columns=target_labels)
estimated_concordance.to_excel(prediction_dir + 'challenger_deep_predict_' + source_name + '_raw' + '.xlsx')

preds[preds >= decision_boundary] = 1
preds[preds < decision_boundary] = 0

estimated_concordance = pd.DataFrame(preds, index=x_labels, columns=target_labels, dtype=int)
estimated_concordance.to_excel(prediction_dir + 'challenger_deep_predict_' + source_name + '_conc' + '.xlsx')

