from __future__ import division, print_function

import random

import numpy as np

from keras import Input
from keras.engine import Model
from keras.layers import Dense, SimpleRNN, TimeDistributed, LSTM

from extkeras.layers.attention import AlexGravesSequenceAttention

# canonical example of attention for alignment

# in this example the model should learn to "parse" through and attended
# sequence and output only relevant parts


def get_training_data(
    n_samples,
    n_labels,
    n_timesteps_attended,
    n_timesteps_labels,
):
    labels = np.random.randint(
        n_labels,
        size=(n_samples, n_timesteps_labels)
    )
    attended_time_idx = range(n_timesteps_attended)
    label_time_idx = range(1, n_timesteps_labels + 1)

    labels_one_hot = np.zeros((n_samples, n_timesteps_labels + 1, n_labels))
    attended = np.zeros((n_samples, n_timesteps_attended, n_labels))
    for i in range(n_samples):
        labels_one_hot[i][label_time_idx, labels[i]] = 1
        positions = sorted(random.sample(attended_time_idx, n_timesteps_labels))
        attended[i][positions, labels[i]] = 1

    return labels_one_hot, attended


n_samples = 10000
n_timesteps_labels = 10
n_timesteps_attended = 30
n_labels = 4

input_labels = Input((n_timesteps_labels, n_labels))
attended = Input((n_timesteps_attended, n_labels))

recurrent_layer = LSTM(units=32, implementation=1)
attention_rnn = AlexGravesSequenceAttention(
    n_components=3,
    recurrent_layer=recurrent_layer,
    return_sequences=True
)
lstm_output = attention_rnn([input_labels, attended])
output_layer = TimeDistributed(Dense(n_labels, activation='softmax'))
output = output_layer(lstm_output)

model = Model(
    inputs=[input_labels, attended],
    outputs=output
)

labels_data, attended_data = get_training_data(
    n_samples,
    n_labels,
    n_timesteps_attended,
    n_timesteps_labels
)
input_labels_data = labels_data[:, :-1, :]
target_labels_data = labels_data[:, 1:, :]

output_data = model.predict([input_labels_data, attended_data])

model.compile(optimizer='Adam', loss='categorical_crossentropy')
model.fit(
    x=[input_labels_data, attended_data],
    y=target_labels_data,
    nb_epoch=5
)
output_data_ = model.predict([input_labels_data, attended_data])
