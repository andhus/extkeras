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
