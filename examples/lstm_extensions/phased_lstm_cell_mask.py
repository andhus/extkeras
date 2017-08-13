from __future__ import division, print_function

import numpy as np
from keras.engine import Input
from keras.engine import Model
from keras.layers import TimeDistributed
from extkeras.layers.recurrent import PhasedLSTMCellMask

n_samples = 3
n_timesteps = 10
n_units = 4

time = Input((n_timesteps, 1))

cell_mask_layer = TimeDistributed(PhasedLSTMCellMask(n_units))
cell_mask = cell_mask_layer(time)
model = Model(inputs=time, outputs=cell_mask)

time_arr = np.arange(n_timesteps).reshape((1, -1, 1)).repeat(n_samples, axis=0)
mask = model.predict(time_arr)

# TODO add plot of mask!
