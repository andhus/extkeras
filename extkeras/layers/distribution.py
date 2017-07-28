from __future__ import division, print_function

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


class ParametersActivationBase(FunctionalBlock):
    # TODO move split params to baseclass...
    pass


class MoGParams1DActivation(FunctionalBlock):

    sub_param_names = ('mu', 'sigma', 'phi')
    n_sub_params = len(sub_param_names)

    def __init__(
        self,
        components,
        mu_activation=None,
        sigma_activation=None,
    ):
        self.components = components
        self.mu_activation = mu_activation or (lambda x: x)
        self.sigma_activation = sigma_activation or ScaledExponential()
        self.phi_activation = softmax

    @classmethod
    def split_params(cls, x):
        dim = x.shape[-1].value
        if not dim % cls.n_sub_params == 0:
            raise ValueError('')
        components = dim // cls.n_sub_params
        mu_ = x[..., :components]
        sigma_ = x[..., components:components * 2]
        phi_ = x[..., components*2:components * 3]

        return mu_, sigma_, phi_

    def __call__(self, x):
        _mu, _sigma, _phi = self.split_params(x)
        mu = self.mu_activation(_mu)
        sigma = self.sigma_activation(_sigma)
        phi = self.phi_activation(_phi)

        return concatenate([mu, sigma, phi], axis=-1)


class MoG1DParams(Dense):

    def __init__(
        self,
        components,
        mu_activation=None,
        sigma_activation=None
    ):
        super(MoG1DParams, self).__init__(
            units=3 * components,
            activation=MoGParams1DActivation(
                components=components,
                mu_activation=mu_activation,
                sigma_activation=sigma_activation,
            )
        )
        self.n_components = components


def neg_log_mog1dpdf(y_true, y_pred):
    """TODO document and check dims of inputs"""
    mu, sigma, phi = MoGParams1DActivation.split_params(y_pred)
    norm = 1. / (np.sqrt(2. * np.pi) * sigma)
    exponent = -(
        K.square(y_true - mu) / (2. * K.square(sigma)) -
        K.log(phi) -
        K.log(norm)
    )
    return -K.logsumexp(exponent, axis=-1)
