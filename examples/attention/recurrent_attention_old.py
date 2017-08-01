from __future__ import print_function, division

import numpy as np

from keras import Input
from keras.engine import Model
from keras.layers import Dense, SimpleRNN

from extkeras.layers.attention_old import RecurrentAttentionWrapper

n_timesteps = 7
n_features = 5
n_samples = 1000

features = Input((n_timesteps, n_features))
attended = Input((n_features, ))  # TODO same as number of features for now due to test hack...

recurrent_layer = SimpleRNN(units=4)
attention_model = Dense(units=4)

rnn = RecurrentAttentionWrapper(
    attention_model=attention_model,
    recurrent_layer=recurrent_layer
)
output_layer = Dense(1, activation='sigmoid')

last_state = rnn(features, attended=attended)
output = output_layer(last_state)

model = Model(
    inputs=[features, attended],
    outputs=output
)

features_data = np.random.randn(n_samples, n_timesteps, n_features)
attended_data = np.ones((n_samples, n_features), dtype=float)
attended_data[::2] = 0.
target_data = attended_data.mean(axis=1, keepdims=True)

output_data = model.predict([features_data, attended_data])

model.compile(optimizer='Adam', loss='binary_crossentropy')
model.fit(x=[features_data, attended_data], y=target_data, nb_epoch=10)
output_data_ = model.predict([features_data, attended_data])
