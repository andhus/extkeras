from __future__ import print_function, division

from keras import backend as K

from extkeras.functional import FunctionalBlock


class ScaleAndShift(FunctionalBlock):

    def __init__(self, scale=1, shift=0):
        self.scale = scale
        self.shift = shift

    def __call__(self, x):
        return self.scale * x + self.shift


class ScaledExponential(FunctionalBlock):

    def __init__(self, scale=1, epsilon=1e-3):
        self.scale = scale
        self.epsilon = epsilon

    def __call__(self, x):
        return self.scale * K.exp(x) + self.epsilon
