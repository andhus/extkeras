from __future__ import division, print_function

import random

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

from keras import backend as K
from keras import Input
from keras.engine import Model
from keras.layers import Dense, TimeDistributed, LSTM
from os import path

from extkeras.layers.attention import GravesSequenceAttention

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
batch_size = 64
n_batches = 1000
units = 32

input_labels = Input((n_timesteps_labels, n_labels))
attended = Input((n_timesteps_attended, n_labels))

recurrent_layer = LSTM(units=units, implementation=1)
attention_rnn = GravesSequenceAttention(
    n_components=3,
    recurrent_layer=recurrent_layer,
    return_sequences=True
)
lstm_output = attention_rnn([input_labels, attended])
attention_output_layer = TimeDistributed(Dense(n_labels, activation='softmax'))
attention_output = attention_output_layer(lstm_output)

attention_model = Model(
    inputs=[input_labels, attended],
    outputs=attention_output
)

labels_data, attended_data = get_training_data(
    n_samples,
    n_labels,
    n_timesteps_attended,
    n_timesteps_labels
)
input_labels_data = labels_data[:, :-1, :]
target_labels_data = labels_data[:, 1:, :]


def sample_batch(batch_size):
    labels_data, attended_data = get_training_data(
        batch_size,
        n_labels,
        n_timesteps_attended,
        n_timesteps_labels
    )
    input_labels_data = labels_data[:, :-1, :]
    target_labels_data = labels_data[:, 1:, :]

    return [input_labels_data, attended_data], target_labels_data


attention_model.compile(optimizer='Adam', loss='categorical_crossentropy')

attention_losses = []
for batch_i in range(n_batches):
    attention_losses.append(
        attention_model.train_on_batch(*sample_batch(batch_size=batch_size))
    )
    if batch_i % 50 == 0 and batch_i > 0:
        print('batch: {}, loss: {}'.format(batch_i, attention_losses[-1]))

# attention_history = attention_model.fit(
#     x=[input_labels_data, attended_data],
#     y=target_labels_data,
#     nb_epoch=5
# )
# attention_output_data = attention_model.predict([input_labels_data, attended_data])


encoder = LSTM(units=units, return_state=True)
[_, attended_enc_h, attended_enc_c] = encoder(attended)
decoder = LSTM(units=units, return_sequences=True)
decoder_output = decoder(input_labels, initial_state=[attended_enc_h, attended_enc_c])
seq_to_seq_output_layer = TimeDistributed(Dense(n_labels, activation='softmax'))
seq_to_seq_output = seq_to_seq_output_layer(decoder_output)

seq_to_seq_model = Model(
    inputs=[input_labels, attended],
    outputs=seq_to_seq_output
)
seq_to_seq_model.compile(optimizer='Adam', loss='categorical_crossentropy')

seq_to_seq_losses = []
for batch_i in range(n_batches):
    seq_to_seq_losses.append(
        seq_to_seq_model.train_on_batch(*sample_batch(batch_size=batch_size))
    )
    if batch_i % 50 == 0 and batch_i > 0:
        print('batch: {}, loss: {}'.format(batch_i, seq_to_seq_losses[-1]))


# seq_to_seq_history = seq_to_seq_model.fit(
#     x=[input_labels_data, attended_data],
#     y=target_labels_data,
#     nb_epoch=5
# )
# seq_to_seq_output_data = seq_to_seq_model.predict([input_labels_data, attended_data])

def n_trainable_params(model):
    return np.sum(
        [K.count_params(p) for p in set(model.trainable_weights)]
    )

f, ax = plt.subplots(1, 1)

df = pd.DataFrame({
    '"Graves" attention (n_params: {})'.format(
        n_trainable_params(attention_model)
    ): attention_losses,
    'Seq-to-Seq (n_params: {})'.format(
        n_trainable_params(seq_to_seq_model)
    ): seq_to_seq_losses,
})
df.to_csv('losses_{}units_len{}-{}.csv'.format(
    units, n_timesteps_labels, n_timesteps_attended)
)

df.plot(ax=ax)

f.suptitle(
    'attention n_timesteps: {}, n_labels: {}'.format(
        n_timesteps_attended, n_timesteps_labels
    )
)
savepath = '/home/andershuss/Data'
f.set_size_inches(15, 9)
f.savefig(
    path.join(
        savepath,
        'losses_{}units_len{}-{}.png'.format(
            units,
            n_timesteps_labels, n_timesteps_attended)
    )
)
