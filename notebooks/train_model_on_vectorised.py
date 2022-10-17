import os
from library.extract_training_datasets import extract_concs_to_sequences
from library.feature_engineering import encode_text_sequences, one_hot_encode_sequences
from utils import create_dir_if_nonexist, read_pickle
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, RepeatVector, Dense
from keras.utils import to_categorical

# Switches
extract_training_data = False

# Paths
work_dir = os.environ['work_dir']
raw_data_dir = work_dir + 'training_concs_hscpc/'

training_data_dir = work_dir + 'training_data/'
create_dir_if_nonexist(training_data_dir)

model_dir = work_dir + 'model/'
create_dir_if_nonexist(model_dir)

# Extract training data
if extract_training_data:
    feature_fd = extract_concs_to_sequences(raw_data_dir, training_data_dir)
else:
    feature_fd = read_pickle(training_data_dir + 'sequence_training_set.pkl')

# Texts-to-sequences
input_text = feature_fd['source_label_sequence'].to_list()
in_seq, in_tokenizer, in_max_words, in_max_len = encode_text_sequences(input_text, max_words_fraction=0.95, pad_position='pre')

output_text = feature_fd['hscpc_sequence'].to_list()
out_seq, out_tokenizer, out_max_words, out_max_len = encode_text_sequences(output_text, max_words_fraction=1.0, pad_position='post')

# One hot encoding
in_seq_ohe = one_hot_encode_sequences(in_seq, in_max_words)
out_seq_ohe = one_hot_encode_sequences(out_seq, in_max_words)

# Define the model
# model = Sequential()
# model.add(LSTM(config.hidden_size, input_shape=(maxlen, len(chars))))
# model.add(RepeatVector(config.digits + 1))
# model.add(LSTM(config.hidden_size, return_sequences=True))
# model.add(TimeDistributed(Dense(len(chars), activation='softmax')))
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# model.summary()
