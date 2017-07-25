from __future__ import print_function, division

import numpy as np

from keras.initializers import Initializer


class TimeGate(Initializer):
    """TODO
    """

    def __init__(
        self,
        period_min=1,
        period_max=100,
        shift_max=10,  # TODO make proportional to period!?
        open_rate=0.05
    ):
        self.period_min = period_min
        self.period_max = period_max
        self.shift_max = shift_max
        self.open_rate = open_rate

    def __call__(self, shape, dtype=None):
        parts, units = shape
        assert parts == 3
        return np.vstack((
            np.random.uniform(self.period_min, self.period_max, units),
            np.random.uniform(0, self.shift_max, units),
            np.zeros(units) + self.open_rate
        ))

    def get_config(self):
        return {
            'period_min': self.period_min,
            'period_max': self.period_max,
            'shift_max': self.shift_max,
            'open_rate': self.open_rate
        }


class AutoTimeGate(TimeGate):

    def __init__(
        self,
        typical_duration=1000,
        typical_n_timesteps=1000,
        open_rate=0.05
    ):
        period_min = typical_duration / typical_n_timesteps
        period_max = typical_duration / min(5., typical_n_timesteps)
        shift_max = period_max  # TODO make proportional
        super(AutoTimeGate, self).__init__(
            period_min=period_min,
            period_max=period_max,
            shift_max=shift_max,
            open_rate=open_rate
        )
