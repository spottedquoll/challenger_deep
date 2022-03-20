import string
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
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


def encode_source_labels(tokenizer, x_labels, max_words):

    # Tokenize the labels
    sequences = tokenizer.texts_to_sequences(x_labels)

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
         'cocoyams', 'tenant', 'fisher']

    return d


def create_tokenizer(source_vocab, max_fraction):

    max_words = int(max_fraction * len(source_vocab))
    tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
    tokenizer.fit_on_texts(source_vocab)

    print('Source vocab length: ' + str(len(source_vocab)) + ', max word setting: ' + str(max_words))

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

    # Should be normalised
    assert np.min(f.reshape(-1)) >= 0 and np.max(f.reshape(-1)) <= 1

    return f