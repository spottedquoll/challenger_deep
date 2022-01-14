from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, Flatten, Dense, LeakyReLU, Dropout, GlobalMaxPooling1D,
                                     BatchNormalization, Conv1D)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from learning import plot_history
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score
from features import augment_by_union

# Switches
augment_training = True
augment_factor = 0.5

# Constants
n_root = 6357
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Paths
work_dir = os.environ['work_dir'] + '/'
save_path = work_dir + '/training_data/'

# Read the vocabularly
source_label_vocab = pd.read_pickle(save_path + 'label_dictionary.pkl')

# Read the training data
fname = save_path + 'hscpc_collected_training_set.pkl'
training_data = pd.read_pickle(fname)
x_labels = training_data['source_row_label'].to_list()
y_labels = training_data['hscpc_labels'].to_list()
position_feature = training_data['position'].values

# Augment
if augment_training:
    x_labels, y_labels, position_feature = augment_by_union(x_labels, y_labels, position_feature, augment_factor)

# Vectorise the labels; by creating a tokenizer, configured to only take into account the max_words
source_vocab = source_label_vocab['source_labels'].to_list()
max_words = int(0.95*len(source_vocab))
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(source_vocab)

print('Source vocab length: ' + str(len(source_vocab)) + ', max word setting: ' + str(max_words))

# This turns strings into lists of integer indices.
sequences = tokenizer.texts_to_sequences(x_labels)
n_samples = len(sequences)
x_features_encoded = np.zeros((n_samples, max_words), dtype=int)

for i, j in enumerate(sequences):
    x_features_encoded[i, j] = 1

# Add extra columns
x_features_encoded = np.append(x_features_encoded, np.expand_dims(position_feature, axis=-1), axis=1)

# One hot encode
y_label_encoded = np.zeros((n_samples, n_root), dtype=int)
for i, j in enumerate(y_labels):
    y_label_encoded[i, j] = 1

# Set feature store name (x)
data = x_features_encoded.copy()

# Shuffle
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
y_label_encoded = y_label_encoded[indices]

# Split the data into a training set and a validation set
n_samples = data.shape[0]
n_train = int(n_samples*0.9)
n_test = n_samples - n_train

x_train = data[:n_train]
y_train = y_label_encoded[:n_train]
x_test = data[n_train: n_train + n_test]
y_test = y_label_encoded[n_train: n_train + n_test]

# Model
model = Sequential()
model.add(Dense(max_words, activation='relu', kernel_regularizer='l2', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
# model.add(Dense(4500, activation='sigmoid', kernel_regularizer='l2', kernel_initializer='he_uniform'))
# model.add(BatchNormalization())
# model.add(Dropout(0.1))
model.add(Dense(6357, activation='sigmoid'))

opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])  # kullback_leibler_divergence
history = model.fit(x_train, y_train, epochs=15, batch_size=400, validation_data=(x_test, y_test))

plot_history(history)

preds = model.predict(x_test)
preds[preds >= 0.5] = 1
preds[preds < 0.5] = 0

acc = accuracy_score(y_test, preds)
print('Score: ' + str(acc))