from __future__ import division, print_function

import abc

import numpy as np

from keras import backend as K
from keras.layers import Dense, concatenate
from keras.activations import softmax

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


class DistributionBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def activation(self, x):
        """apply activation function on input tensor"""
        pass

    @abc.abstractmethod
    def loss(self, y_true, y_pred):
        """Implementation of standard loss for this this distribution normally
        -log(pdf(y_true)) where pdf is parameterized by y_pred
        """
        pass

    @abc.abstractproperty
    def n_params(self):
        pass


class MixtureDistributionBase(DistributionBase):
    __metaclass__ = abc.ABCMeta

    n_param_types = None

    def __init__(self, n_components):
        self.n_components = n_components

    @property
    def mixture_weight_activation(self):
        return softmax

    @classmethod
    def split_param_types(cls, x):
        """Splits input tensor into the different param types.
        Assumes same number of parameters for each param type...
        """
        # TODO use n_components instead?
        dim = x.shape[-1].value
        if not dim % cls.n_param_types == 0:
            raise ValueError(
                'this activation must be given tensor with last dimension'
                'even dividable with number of parameter types: {}'.format(
                    cls.n_param_types
                )
            )
        components = dim // cls.n_param_types
        param_types = [
            x[..., i*components:(i+1)*components]
            for i in range(cls.n_param_types)
        ]

        return param_types

    @property
    def n_params(self):
        return self.n_param_types * self.n_components


class MixtureOfGaussian1D(MixtureDistributionBase):

    param_type_names = ('mixture_weight', 'mu', 'sigma')
    n_param_types = len(param_type_names)

    def __init__(
        self,
        n_components,
        mu_activation=None,
        sigma_activation=None,
    ):
        super(MixtureOfGaussian1D, self).__init__(n_components)
        self.mu_activation = mu_activation or (lambda x: x)
        self.sigma_activation = sigma_activation or ScaledExponential()

    def activation(self, x):
        _mixture_weights, _mu, _sigma = self.split_param_types(x)
        mixture_weights = self.mixture_weight_activation(_mixture_weights)
        mu = self.mu_activation(_mu)
        sigma = self.sigma_activation(_sigma)

        return concatenate([mixture_weights, mu, sigma], axis=-1)

    @classmethod
    def loss(cls, y_true, y_pred):
        """TODO document and check dims of inputs
        """
        mixture_weights, mu, sigma, = cls.split_param_types(y_pred)
        norm = 1. / (np.sqrt(2. * np.pi) * sigma)
        exponent = -(
            K.square(y_true - mu) / (2. * K.square(sigma)) -
            K.log(mixture_weights) -
            K.log(norm)
        )
        return -K.logsumexp(exponent, axis=-1)


class DistributionOutputLayer(Dense):

    def __init__(self, distribution, **kwargs):
        self.distribution = distribution
        if 'units' in kwargs or 'activation' in kwargs:
            raise ValueError('')  # TODO
        super(DistributionOutputLayer, self).__init__(
            units=distribution.n_params,
            activation=distribution.activation
        )
