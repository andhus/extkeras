from __future__ import print_function, division

import numpy as np
from keras import Input
from keras.engine import Model
from keras.layers import Lambda, TimeDistributed

from extkeras.layers.distribution import MoG1DParams, neg_log_mog1dpdf

n_timesteps = 3
n_features = 5
n_samples = 1
n_components = 2

features = Input((n_timesteps, n_features), name='features')
target = Input((n_timesteps, 1), name='target')

params_layer = TimeDistributed(MoG1DParams(components=n_components))
params = params_layer(features)

loss_layer = Lambda(lambda y_true_pred: neg_log_mog1dpdf(*y_true_pred))
loss = loss_layer([target, params])

params_and_loss_model = Model(inputs=[target, features], outputs=[params, loss])

features_data = np.random.randn(n_samples, n_timesteps, n_features)
target_data = np.random.rand(n_samples, n_timesteps, 1)

[mog_params_data, loss_data] = params_and_loss_model.predict(
    [target_data, features_data]
)
