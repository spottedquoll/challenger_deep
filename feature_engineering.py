import string
import random
import numpy as np


def create_position_feature(n_source, n_root):
    """
        The feature positions each row in the context of the whole concordance
        Absolute position and absolute number of rows are also generated.
        These are normalised wrt to n_root categories, to stop them growing without bounds
    """

    assert n_source > 0

    features = []

    for i in range(n_source):
        features.append([i/n_root, n_source/n_root, (i + 1) / n_source])

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


def augment_by_union(x, y, position_feature, factor):

    n_real_records = len(x)
    extra_rows = int(n_real_records*factor)

    for i in range(extra_rows):

        rand_idx = random.randint(0, n_real_records-2)
        new_x_label = x[rand_idx] + ' ' + x[rand_idx+1]
        new_y_label = np.unique(np.concatenate((y[rand_idx], y[rand_idx+1])))
        new_position = np.mean([position_feature[rand_idx][2], position_feature[rand_idx+1][2]])

        x.append(new_x_label)
        y.append(new_y_label)
        position_feature.append([position_feature[rand_idx][0], position_feature[rand_idx][1], new_position])

    print('Training x augmented from ' + str(n_real_records) + ' to ' + str(len(x)) + ' records')

    return x, y, position_feature


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
         'cocoyams']

    return d

