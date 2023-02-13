import string
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from fuzzywuzzy import fuzz


def create_position_feature(x_labels):
    """
        The feature positions each row in the context of the whole concordance
        Absolute position and absolute number of rows are also generated.
    """

    n_source = len(x_labels)
    assert n_source > 0

    features = []

    for i, v in enumerate(x_labels):
        features.append([(i + 1) / n_source])

    return features


def one_hot_encode_source_labels(sequences, x_labels, max_words):

    # Create store for enconded labels
    n_samples = len(x_labels)
    x_features_encoded = np.zeros((n_samples, max_words), dtype=int)

    # One hot encode
    for i, j in enumerate(sequences):
        if j is not None and j != []:
            assert min(j) > 0 and max(j) <= max_words
            j_ar = np.array(j) - 1
            x_features_encoded[i, j_ar] = 1
        else:
            print('No tokens available for: ' + x_labels[i])

    return x_features_encoded


def clean_text_label(txt):

    # Remove punctuation and make lower case
    lbl = txt.lower()
    for c in string.punctuation:
        lbl = lbl.replace(c, " ")

    # Remove numeric characters
    lbl = ''.join([i for i in lbl if not i.isdigit()])

    # Remove double whitespace
    lbl = lbl.replace('   ', ' ')
    lbl = lbl.replace('  ', ' ')

    return lbl


def dictionary_supplement():

    d = ['millwork', 'breweries', 'brewery', 'wineries', 'distilleries', 'retailers', 'owner', 'occupied', 'housing',
         'cocoyams', 'tenant', 'fisher', 'prints', 'cropping']

    return d


def create_tokenizer(source_vocab, max_fraction):

    max_words = int(max_fraction * len(source_vocab))
    tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+-/:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
    tokenizer.fit_on_texts(source_vocab)

    return tokenizer, max_words


def make_c100_features(source_labels, c100_labels):
    """
        Maps the string similarity between each source label and each c100 label
    """

    f = np.zeros((len(source_labels), len(c100_labels)))
    whitelist = string.ascii_letters + string.digits + ' '

    for i, r in enumerate(source_labels):
        r_clean = ''.join(c for c in r if c in whitelist)  # clean the string
        for j, s in enumerate(c100_labels):
            s_clean = ''.join(c for c in s if c in whitelist)  # clean the string
            score = fuzz.token_sort_ratio(r_clean, s_clean) * (1/100)
            assert np.isfinite(score)
            f[i, j] = score

    # Ensure normalisation
    assert np.min(f.reshape(-1)) >= 0 and np.max(f.reshape(-1)) <= 1

    return f


def encode_x_labels(tokenizer, x_labels, max_words, one_hot_encoding=True, max_len=None):

    # Tokenize the labels
    sequences = tokenizer.texts_to_sequences(x_labels)

    if one_hot_encoding:
        x_features_encoded = one_hot_encode_source_labels(sequences, x_labels, max_words)
        return x_features_encoded, None
    else:
        if max_len is None:
            max_len = max([len(x) for x in sequences])
        x_features_encoded = pad_sequences(sequences, padding='post', truncating='post', maxlen=max_len)

        return x_features_encoded, max_len


def encode_text_sequences(texts, max_words_fraction=0.95, pad=True, pad_position='post', max_len=None):

    # Tests
    assert 0 < max_words_fraction <= 1
    assert pad_position in ['pre', 'post']

    # Create tokenizer (limited to max_words)
    tokenizer, max_words = create_tokenizer(texts, max_words_fraction)

    # Tokenize the labels
    sequences = tokenizer.texts_to_sequences(texts)

    if pad:
        if max_len is None:
            max_len = max([len(x) for x in sequences])
        sequences = pad_sequences(sequences, padding=pad_position, truncating=pad_position, maxlen=max_len)

    return sequences, tokenizer, max_words, max_len


def one_hot_encode_sequences(sequences, max_words):

    # Create store for enconded labels
    n_samples = sequences.shape[0]
    x_features_encoded = np.zeros((n_samples, max_words), dtype=int)

    # One hot encode
    for i, j in enumerate(sequences):
        if j is not None and j != []:
            assert min(j) >= 0 and max(j) <= max_words
            j_ar = np.array(j) - 1
            x_features_encoded[i, j_ar] = 1
        else:
            print('No tokens available for sample ' + str(i))

    return x_features_encoded