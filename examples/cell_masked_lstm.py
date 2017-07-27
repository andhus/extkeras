from __future__ import print_function, division

import numpy as np
from keras import Input
from keras.engine import Model

from extkeras.layers.recurrent import CellMaskedLSTM

n_units = 2

features_and_cell_mask = Input((5, 5))
cell_masked_lstm = CellMaskedLSTM(
    units=n_units,
    output_cells=True,  # to be able to inspect that the cells are masked
    return_sequences=True
)
state_cell_sequence = cell_masked_lstm(features_and_cell_mask)
model = Model(inputs=features_and_cell_mask, outputs=state_cell_sequence)

features_data = np.array([
    [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ]
])
cell_mask_data = np.array([
    [
        [0., 0.],
        [0., 0.],
        [1., 1.],
        [1., 0.],
        [0., 1.],
    ]
])
feature_and_cell_mask_data = np.concatenate([features_data, cell_mask_data], 2)

state_cell_sequence_data = model.predict(feature_and_cell_mask_data)
cell_sequence_sample = state_cell_sequence_data[0, :, n_units:]
assert (cell_sequence_sample[:2] == 0).all()
assert cell_sequence_sample[3, 1] == cell_sequence_sample[2, 1]
assert cell_sequence_sample[4, 0] == cell_sequence_sample[3, 0]
