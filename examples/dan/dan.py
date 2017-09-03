from __future__ import division, print_function


import numpy as np
from keras import Input
from keras.layers import Dense, TimeDistributed
from keras.models import Model

from extkeras.layers.dan import DANRecurrent


n_samples = 1000
batch_size = 32
input_size = 3
units = 10
key_size = 10
nodes = 100
T = 4


dan = DANRecurrent(
    units=units,
    key_size=key_size,
    nodes=nodes,
    return_sequences=True
)

input_ = Input((input_size,))
initial_value = Input((units,))
mock_sequence = Input((T, units))

dense = Dense(units=key_size, activation='softmax')
initial_key = dense(input_)
dan_output_sequence = dan(mock_sequence, initial_state=[initial_key, initial_value])
output_layer = TimeDistributed(Dense(units=1, activation=None))
output_sequence = output_layer(dan_output_sequence)

model = Model(
    inputs=[mock_sequence, input_, initial_value],
    outputs=output_sequence
)
input_data = np.random.randn(n_samples, input_size)
target_data = input_data.sum(axis=-1, keepdims=True)[:, None, :].repeat(T, axis=1)

initial_value_data = np.zeros((n_samples, units))
mock_sequence_data = np.zeros((n_samples, T, units))

batch = [
    mock_sequence_data[:batch_size],
    input_data[:batch_size],
    initial_value_data[:batch_size]
]

output_sequence_data = model.predict(batch)
model.compile(optimizer='Adam', loss='mse')

model.fit(
    x=[mock_sequence_data, input_data, initial_value_data],
    y=target_data,
    epochs=100
)

output_sequence_data_ = model.predict(
    [mock_sequence_data, input_data, initial_value_data]
)
