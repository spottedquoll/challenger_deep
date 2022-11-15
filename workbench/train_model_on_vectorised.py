import os
import numpy as np
from library.extract_training_datasets import extract_concs_to_sequences
from library.feature_engineering import encode_text_sequences, one_hot_encode_sequences
from utils import create_dir_if_nonexist, read_pickle
from keras.layers import LSTM, Dense
from keras import Input, Model

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

n_samples = feature_fd.shape[0]
del feature_fd

# One hot encoding
# in_seq_ohe = one_hot_encode_sequences(in_seq, in_max_words)
# out_seq_ohe = one_hot_encode_sequences(out_seq, in_max_words)

n_encoder_tokens = in_tokenizer.num_words
encoder_input_data = np.zeros((n_samples, in_max_len, n_encoder_tokens), dtype="float32")

n_decoder_tokens = out_tokenizer.num_words
decoder_input_data = np.zeros((n_samples, out_max_len, n_decoder_tokens), dtype="float32")
decoder_target_data = np.zeros((n_samples, out_max_len, n_decoder_tokens), dtype="float32")

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0


# Config
batch_size = 25  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 128  # Latent dimensionality of the encoding space.

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, n_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, n_decoder_tokens))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(n_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)
