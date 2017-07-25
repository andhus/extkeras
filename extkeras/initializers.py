from __future__ import print_function, division

from keras import backend as K
from keras.initializers import Initializer


# TODO
# class Range(Initializer):
#     """Initializer that generates tensors initialized to a constant value.
#
#     # Arguments
#         value: float; the value of the generator tensors.
#     """
#
#     def __init__(self, value=0):
#         self.value = value
#
#     def __call__(self, shape, dtype=None):
#         return K.constant(self.value, shape=shape, dtype=dtype)
#
#     def get_config(self):
#         return {'value': self.value}
