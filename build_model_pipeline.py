from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, LeakyReLU, Dropout, GlobalMaxPooling1D, BatchNormalization, Conv1D)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score
from feature_engineering import augment_by_union, encode_source_labels, create_position_feature
from datetime import date
from utils import write_pickle

print('Building predictive model')

# Switches
augment_training = True
augment_factor = 0.5
add_position_features = True

# Constants
n_root = 6357
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
date_str_today = date.today().strftime("%Y-%m-%d")

# Paths
work_dir = os.environ['work_dir'] + '/'
save_path = work_dir + '/training_data/'

# Read the vocabularly
source_label_vocab = pd.read_pickle(save_path + 'label_dictionary.pkl')

# Read the training x
fname = save_path + 'hscpc_collected_training_set.pkl'
training_data = pd.read_pickle(fname)
x_labels = training_data['source_row_label'].to_list()
y_labels = training_data['hscpc_labels'].to_list()
position_feature = create_position_feature(len(x_labels), n_root)

n_samples = len(x_labels)

# Tests; check input data
max_hscpc_index = [max(s) for s in zip(*y_labels)]
min_hscpc_index = [min(s) for s in zip(*y_labels)]
assert max_hscpc_index[0] == n_root-1 and min_hscpc_index[0] == 0

# Augment
if augment_training:
    x_labels, y_labels, position_feature = augment_by_union(x_labels, y_labels, position_feature, augment_factor)
    n_samples = len(x_labels)

# Create tokenizer, configured to only take into account the max_words
source_vocab = source_label_vocab['source_labels'].to_list()
max_words = int(0.98*len(source_vocab))
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
model = Sequential()
model.add(Dense(max_words, activation='relu', kernel_regularizer='l2', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(6357, activation='sigmoid'))

opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc', 'categorical_accuracy'])  # kullback_leibler_divergence
history = model.fit(x_train, y_train, epochs=15, batch_size=600, validation_data=(x_test, y_test))

#plot_history(history)

preds = model.predict(x_test)
preds[preds >= 0.5] = 1
preds[preds < 0.5] = 0

acc = accuracy_score(y_test, preds)
print('Score: ' + str(acc))

# Retrain model with full training set
model.fit(x, y, epochs=15, batch_size=600, validation_data=(x, y))

preds = model.predict(x)
decision_boundary = 0.5
preds[preds >= decision_boundary] = 1
preds[preds < decision_boundary] = 0

acc = accuracy_score(y, preds)
print('Final score: ' + str(acc))

# Save model object
model_fname = work_dir + 'model/' + 'model_' + date_str_today + '_w' + str(max_words) + '_s' + str(n_samples) + '.pkl'
write_pickle(model_fname, model)

print('Saved model to disk')

# Save feature meta
fname_feature_meta = work_dir + 'model/' + 'feature_meta_' + date_str_today + '_w' + str(max_words) + '.pkl'
write_pickle(fname_feature_meta, feature_meta)

print('Saved feature meta to disk')

print('Finished model build')