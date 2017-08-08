from __future__ import print_function, division

import numpy as np

from keras import Input
from keras.engine import Model
from keras.layers import Dense, SimpleRNN, TimeDistributed

from extkeras.layers.attention import DenseStatelessAttention

n_timesteps = 7
n_features = 5
n_features_attention = 2
n_samples = 1000

features = Input((n_timesteps, n_features))
attended = Input((n_features_attention, ))

recurrent_layer = SimpleRNN(units=4, implementation=1)
attention_rnn = DenseStatelessAttention(
    units=3,
    recurrent_layer=recurrent_layer
)
output_layer = Dense(1, activation='sigmoid')

last_state = attention_rnn([features, attended])
output = output_layer(last_state)

model = Model(
    inputs=[features, attended],
    outputs=output
)

features_data = np.random.randn(n_samples, n_timesteps, n_features)
attended_data = np.ones((n_samples, n_features_attention), dtype=float)
attended_data[::2] = 0.
target_data = attended_data.mean(axis=1, keepdims=True)

output_data = model.predict([features_data, attended_data])

model.compile(optimizer='Adam', loss='binary_crossentropy')
model.fit(x=[features_data, attended_data], y=target_data, nb_epoch=20)
output_data_ = model.predict([features_data, attended_data])
