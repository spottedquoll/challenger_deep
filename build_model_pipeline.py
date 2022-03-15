from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score
from library.feature_engineering import encode_source_labels, create_position_feature
from library.augmentation import augment_by_adjacent_union
from datetime import date
from utils import write_pickle, create_dir_if_nonexist
from library.training_toolkit import extract_concordances, create_source_label_vocabularly
from library.model_configs import embedded_conv, basic_dense, basic_dense_plus


print('Building predictive model')

# Switches
extract_training_data = False
rebuild_source_vocabularly = False
augment_training = True
augment_factor = 2.0
add_position_features = True
decision_boundary = 0.90
n_epochs = 30
n_batch_size = 600
alpha = 0.00001  # learning rate

# Constants
n_root = 6357
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
date_str_today = date.today().strftime("%Y-%m-%d")

# Paths
work_dir = os.environ['work_dir']
raw_data_dir = work_dir + 'training_concs_hscpc/'
training_data_dir = work_dir + 'training_data/'

create_dir_if_nonexist(training_data_dir)

# Extract training data
if extract_training_data:
    extract_concordances(raw_data_dir, training_data_dir)

# Construct the vocabulary from the source data
if rebuild_source_vocabularly:
    create_source_label_vocabularly(raw_data_dir, training_data_dir, n_root)

# Read the vocabularly
source_label_vocab = pd.read_pickle(training_data_dir + 'label_dictionary.pkl')

# Read and wrangle the training data
training_data = pd.read_pickle(training_data_dir + 'hscpc_collected_training_set.pkl')

x_labels = training_data['source_row_label'].to_list()
y_labels = training_data['hscpc_labels'].to_list()
n_samples = len(x_labels)

# Append extra positional features
position_feature = create_position_feature(len(x_labels), n_root)

# Tests; check input data
max_hscpc_index = [max(s) for s in zip(*y_labels)]
min_hscpc_index = [min(s) for s in zip(*y_labels)]
assert max_hscpc_index[0] == n_root-1 and min_hscpc_index[0] == 0

# Augment
if augment_training:
    x_labels, y_labels, position_feature = augment_by_adjacent_union(x_labels, y_labels, position_feature, augment_factor)
    n_samples = len(x_labels)

# Create tokenizer, configured to only take into account the max_words
source_vocab = source_label_vocab['source_labels'].to_list()
max_words = int(0.99*len(source_vocab))
tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
tokenizer.fit_on_texts(source_vocab)

print('Source vocab length: ' + str(len(source_vocab)) + ', max word setting: ' + str(max_words))

feature_meta = {'tokenizer': tokenizer, 'max_words': max_words}

# One-hot-encode x labels
x_features_encoded = encode_source_labels(tokenizer, x_labels, max_words)

# Add extra features
if add_position_features:
    x_features_encoded = np.append(x_features_encoded, np.array(position_feature), axis=1)
feature_meta['add_position_features'] = add_position_features

# One hot encode
y = np.zeros((n_samples, n_root), dtype=int)
for i, j in enumerate(y_labels):
    y[i, j] = 1

# Set feature store name (x)
x = x_features_encoded.copy()

# Shuffle
indices = np.arange(x.shape[0])
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Split the x into a training set and a validation set
n_samples = x.shape[0]
n_train = int(n_samples*0.9)
n_test = n_samples - n_train

x_train = x[:n_train]
y_train = y[:n_train]
x_test = x[n_train: n_train + n_test]
y_test = y[n_train: n_train + n_test]

# Model
#model = embedded_conv(max_words+1, x.shape[1], n_root)
model = basic_dense(x.shape[1], n_root)
#model = basic_dense_plus(x.shape[1], n_root)
opt = Adam(learning_rate=alpha)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc', 'categorical_accuracy'])
history = model.fit(x_train, y_train, epochs=n_epochs, batch_size=n_batch_size, validation_data=(x_test, y_test))

preds = model.predict(x_test)
preds[preds >= decision_boundary] = 1
preds[preds < decision_boundary] = 0

acc = accuracy_score(y_test, preds)
sse = np.sum(np.square(preds - y_test))
rmse = np.sqrt(np.mean(np.square(preds - y_test)))

print('Accuracy: ' + "{:.4f}".format(acc) + ', SSE: ' + str(sse) + ', RMSE: ' + "{:.3f}".format(rmse))

# Retrain model with full training set (should the batch size be reset here?)
model.fit(x, y, epochs=n_epochs, batch_size=n_batch_size, validation_data=(x, y))

preds = model.predict(x)
preds[preds >= decision_boundary] = 1
preds[preds < decision_boundary] = 0

acc = accuracy_score(y, preds)
sse = np.sum(np.square(preds - y))
rmse = np.sqrt(np.mean(np.square(preds - y)))

print('Final score: accuracy: ' + "{:.4f}".format(acc) + ', SSD: ' + str(sse) + ', RMSE: ' + "{:.3f}".format(rmse))

# Save model object
n_layers = len(model.layers)
model_meta_name = 'model_' + date_str_today + '_w' + str(max_words) + '_s' + str(n_samples) + '_l' + str(n_layers)
model_fname = work_dir + 'model/' + model_meta_name + '.pkl'
write_pickle(model_fname, model)

print('Saved model to disk. model_meta: ' + model_meta_name)

# Save feature meta
feature_meta_name = 'feature_meta_' + date_str_today + '_w' + str(max_words)
fname_feature_meta = work_dir + 'model/' + feature_meta_name + '.pkl'
write_pickle(fname_feature_meta, feature_meta)

print('Saved feature meta to disk. feature_meta: ' + feature_meta_name)

print('Finished model build')