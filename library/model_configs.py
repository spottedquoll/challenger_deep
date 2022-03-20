from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, Flatten, Embedding)
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.constraints import max_norm


def embedded_conv(embed_dim, feature_dim, output_dim):

    model = Sequential()
    model.add(Embedding(embed_dim, 64, input_length=feature_dim))
    model.add(Conv1D(128, 3, padding="same", activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(round(output_dim*0.75), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(output_dim, activation='sigmoid'))

    return model


def basic_dense(feature_dim, output_dim):

    model = Sequential()
    model.add(Dense(feature_dim, activation='relu', kernel_regularizer='l2', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(output_dim, activation='sigmoid'))

    return model


def basic_dense_plus(feature_dim, output_dim):

    model = Sequential()
    model.add(Dense(feature_dim, activation='relu', kernel_regularizer='l2', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(output_dim*1.5, activation='relu', kernel_constraint=max_norm(2.)))
    model.add(Dropout(0.1))
    model.add(Dense(output_dim, activation='softmax'))

    return model