from __future__ import print_function, division

import numpy as np
from keras import Input
from keras.engine import Model

from extkeras.layers.recurrent import PhasedLSTM

n_timesteps = 7
n_features = 5
n_samples = 3

features = Input((n_timesteps, n_features))
time = Input((n_timesteps, 1))

phased_lstm = PhasedLSTM(units=4, return_sequences=True)
state_sequence = phased_lstm([features, time])

model = Model(
    inputs=[features, time],
    outputs=state_sequence
)

features_data = np.random.randn(n_samples, n_timesteps, n_features)
time_data = np.arange(n_timesteps).reshape((1, -1, 1)).repeat(n_samples, axis=0)
state_sequence_data = model.predict([features_data, time_data])
