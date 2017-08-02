from __future__ import print_function, division

import numpy as np
from keras import Input
from keras.engine import Model
from keras.layers import TimeDistributed
from numpy.testing import assert_almost_equal

from extkeras.layers.distribution import (
    MixtureOfGaussian1D,
    DistributionOutputLayer
)

n_timesteps = 100
n_features = 5
n_samples = 1000
n_components = 2

features = Input((n_timesteps, n_features), name='features')
target = Input((n_timesteps, 1), name='target')

distribution = MixtureOfGaussian1D(n_components=n_components)

params_layer = TimeDistributed(DistributionOutputLayer(distribution))
params = params_layer(features)


model = Model(inputs=features, outputs=params)

features_data = np.abs(np.random.randn(n_samples, n_timesteps, n_features))
features_sum = features_data.sum(axis=-1, keepdims=True)
random_sign = (np.random.random(features_sum.shape) > 0.7).astype(int) * 2 - 1
random_addition = np.random.randn(*features_sum.shape)
target_data = features_sum * random_sign + random_addition

model.compile(optimizer='rmsprop', loss=distribution.loss)
model.fit(x=features_data, y=target_data, nb_epoch=150, batch_size=100)
params_data = model.predict(features_data)

# check convergence... > 100 epochs needed for full convergence
mixture_weight, mu, sigma = distribution.split_param_types(params_data[0, 0])
assert_almost_equal(sorted(mixture_weight), [0.3, 0.7], decimal=1)
assert_almost_equal(sigma, [1.0, 1.0], decimal=1)
