from __future__ import print_function, division

import numpy as np

from keras.models import Sequential
from extkeras.layers.recurrent import CellMaskedLSTM

x = np.array([
    [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]
])

cell_mask = np.array([
    [
        [0., 0.],
        [0., 0.],
        [1., 1.],
        [1., 0.],
        [0., 1.],
    ]
])

x_cell_mask = np.concatenate([x, cell_mask], 2)

model = Sequential()
model.add(
    CellMaskedLSTM(
        units=2,
        input_shape=(x.shape[1], x.shape[2] + 2),
        return_sequences=True
    )
)

out = model.predict(x_cell_mask)



# features = Input((sequence_length, input_dim))
# time = Input((sequence_length, 1))
#
# cell_mask_layer = TimeDistributed(PhasedLSTMCellMask(state_dim))
# cell_mask = cell_mask_layer(time)
#
# features_and_cell_mask = merge(
#     [features, cell_mask],
#     mode='concat',
#     concat_axis=2
# )
# lstm = CellMaskedLSTM(state_dim, return_sequences=True)
# state_sequence = lstm(features_and_cell_mask)
#
# model = Model(
#     input=[features, time],
#     output=state_sequence
# )
#
# features_arr = np.random.randn(n_samples, sequence_length, input_dim)
# time_arr = np.arange(sequence_length).reshape((1, -1, 1)).repeat(n_samples, axis=0)
# state_sequence_arr = model.predict([features_arr, time_arr])

