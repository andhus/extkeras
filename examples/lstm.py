from __future__ import print_function, division

import numpy as np
from keras import Input
from keras.engine import Model

from extkeras.layers.recurrent import LSTM

n_timesteps = 7
n_features = 5
n_samples = 3
n_units = 4

features = Input((n_timesteps, n_features))

lstm = LSTM(units=n_units, return_sequences=True)
state_sequence = lstm(features)
lstm_model = Model(inputs=features, outputs=state_sequence)

lstm_cellout = LSTM(units=n_units, output_cells=True, return_sequences=True)
state_cell_sequence = lstm_cellout(features)
lstm_cellout_model = Model(inputs=features, outputs=state_cell_sequence)

features_data = np.random.randn(n_samples, n_timesteps, n_features)

state_sequence_data = lstm_model.predict(features_data)
state_cell_sequence_data = lstm_cellout_model.predict(features_data)

assert state_sequence_data.shape[2] == n_units
assert state_cell_sequence_data.shape[2] == n_units * 2
