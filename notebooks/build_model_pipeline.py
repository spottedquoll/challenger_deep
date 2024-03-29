from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from library.feature_engineering import encode_x_labels, create_tokenizer, make_c100_features
from library.augmentation import augment_by_adjacent_union
from datetime import date
from utils import write_pickle, create_dir_if_nonexist, read_pickle
from library.training_toolkit import extract_concordances_into_rows, create_source_label_vocabularly
from library.model_configs import embedded_conv, basic_dense, basic_dense_plus


print('Building predictive model')

# Switches
use_prepared_data = False
extract_training_data = False
rebuild_source_vocabularly = False
augment_training = False
add_position_features = True
add_isic_100_features = True
x_feature_one_hot_encoding = True

# Settings
augment_factor = 2
test_set_size = 0.1
max_vocab_fraction = 0.99
decision_boundary = 0.90
n_epochs = 30
n_batch_size = 900
alpha = 0.0001  # learning rate

# Constants
n_root = 6357
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
date_str_today = date.today().strftime("%Y-%m-%d")

# Paths
work_dir = os.environ['work_dir']
raw_data_dir = work_dir + 'training_concs_hscpc/'

training_data_dir = work_dir + 'training_data/'
create_dir_if_nonexist(training_data_dir)

model_dir = work_dir + 'model/'
create_dir_if_nonexist(model_dir)

fname_prepared_data = work_dir + 'training_data/' + 'prepared_data_aug' + str(augment_training) + '.pkl'
if use_prepared_data:

    prepared_data = read_pickle(fname_prepared_data)
    x = prepared_data['x']
    y = prepared_data['y']
    tokenizer = prepared_data['tokenizer']
    max_words = prepared_data['max_words']
    sequences = prepared_data['sequences']

else:

    # Extract training data
    if extract_training_data:
        extract_concordances_into_rows(raw_data_dir, training_data_dir)

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

    # Tests; check input data
    max_hscpc_index = [max(s) for s in zip(*y_labels)]
    min_hscpc_index = [min(s) for s in zip(*y_labels)]
    assert max_hscpc_index[0] == n_root-1 and min_hscpc_index[0] == 0

    # One hot encode y
    y = np.zeros((n_samples, n_root), dtype=int)
    for i, j in enumerate(y_labels):
        y[i, j] = 1

    # Create tokenizer (limited to max_words)
    source_vocab = source_label_vocab['source_labels'].to_list()
    tokenizer, max_words = create_tokenizer(source_vocab, max_vocab_fraction)

    # Encode x labels
    x_features_encoded = encode_x_labels(tokenizer, x_labels, max_words, one_hot_encoding=x_feature_one_hot_encoding)

    # Add label position feature
    if add_position_features:
        x_features_encoded = np.hstack((x_features_encoded, training_data['position'].to_numpy().reshape(-1, 1)))

    # C100 features
    if add_isic_100_features:
        print('Calculating string similarity between x labels and C100...')
        c100_labels = pd.read_excel(work_dir + 'hscpc/c100_labels.xlsx')['sector'].to_list()
        new_features = make_c100_features(training_data['source_row_label'].to_list(), c100_labels)
        x_features_encoded = np.hstack((x_features_encoded, new_features))

    # Augment (augment 'real' training data with extra generated samples)
    if augment_training:
        x_features_encoded, y = augment_by_adjacent_union(x_features_encoded, y, max_words, augment_factor)

    # Final feature matrix
    x = x_features_encoded.copy()

    # Save prepared dataset
    prepared_data = {'x': x, 'y': y, 'tokenizer': tokenizer, 'max_words': max_words, 'sequences': sequences,
                     'x_feature_one_hot_encoding': x_feature_one_hot_encoding}

    write_pickle(fname_prepared_data, prepared_data)

# Training set properties
n_features = x.shape[1]
n_samples = x.shape[0]

print('Training set contains ' + str(n_features) + ' features and ' + str(x.shape[0]) + ' records')

# Test-train split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_set_size)

# Model
# model = basic_dense_plus(n_features, n_root)
model = basic_dense(n_features, n_root)
opt = Adam(learning_rate=alpha)
callback = EarlyStopping(monitor='acc', patience=3)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc', 'categorical_accuracy',
                                                                       'mean_squared_error'])

history = model.fit(x_train, y_train, epochs=n_epochs, batch_size=n_batch_size, validation_data=(x_test, y_test),
                    callbacks=[callback])

preds = model.predict(x_test)
preds[preds >= decision_boundary] = 1
preds[preds < decision_boundary] = 0

acc = accuracy_score(y_test, preds)
sse = np.sum(np.square(preds - y_test))
rmse = np.sqrt(np.mean(np.square(preds - y_test)))

print('Accuracy: ' + "{:.4f}".format(acc) + ', SSE: ' + str(sse) + ', RMSE: ' + "{:.3f}".format(rmse))

# Retrain model with full training set
model.fit(x, y, epochs=n_epochs, batch_size=n_batch_size, validation_data=(x, y), callbacks=[callback])

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
model_fname = model_dir + model_meta_name + '.pkl'
write_pickle(model_fname, model)

print('Saved model to disk. model_meta: ' + model_meta_name)

# Save feature meta
feature_meta = {'tokenizer': tokenizer, 'max_words': max_words, 'add_position_features': add_position_features,
                'sequences': sequences, 'add_isic_100_features': add_isic_100_features,
                'x_feature_one_hot_encoding': x_feature_one_hot_encoding}

feature_meta_name = 'feature_meta_' + date_str_today + '_w' + str(max_words)
fname_feature_meta = work_dir + 'model/' + feature_meta_name + '.pkl'
write_pickle(fname_feature_meta, feature_meta)

print('Saved feature meta to disk. feature_meta: ' + feature_meta_name)

print('Finished model build')