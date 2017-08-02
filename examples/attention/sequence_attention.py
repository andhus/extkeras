from __future__ import print_function, division

import numpy as np

from keras import backend as K
from keras import Input
from keras.engine import Model
from keras.layers import Lambda

from extkeras.layers.distribution import (
    MixtureOfGaussian1D,
    DistributionOutputLayer
)


n_samples = 2
n_features = 5

attended_len = 7
attended_n_features = 2

features = Input((n_features, ))
attended = Input((attended_len, attended_n_features))


distribution = MixtureOfGaussian1D(n_components=3)
output_layer = DistributionOutputLayer(distribution)

params = output_layer(features)


def get_attention_h(inputs):
    [params_, attended_] = inputs
    att_idx = K.constant(np.arange(attended_len)[None, :, None])

    mw, mu, sigma = distribution.split_param_types(params_)

    mu = K.expand_dims(mu, 1)
    sigma = K.expand_dims(sigma, 1)
    mw = K.expand_dims(mw, 1)

    norm = 1. / (np.sqrt(2. * np.pi) * sigma)
    attention_w = K.sum(
        mw * norm * K.exp(
            - K.square(mu - att_idx) / (2 * K.square(sigma))
        ),
        axis=-1
    )
    attention_w = K.expand_dims(attention_w, -1)

    attention_h = K.sum(attention_w * attended_, axis=1)

    return attention_h

attention_h = Lambda(get_attention_h)([params, attended])

features_data = np.random.randn(n_samples, n_features)
attended_data = np.random.randn(n_samples, attended_len, attended_n_features)

model = Model(inputs=[features, attended], outputs=attention_h)
attention_h_data = model.predict([features_data, attended_data])

# sess = K.get_session()
# [mw_, mu_, sigma_, attention_w_, attention_h_] = sess.run(
#     [mw, mu, sigma, attention_w, attention_h],
#     feed_dict={features: features_data, attended: attended_data}
# )

# model = Model(inputs=features, outputs=params)
#
#
# params_data = model.predict(features_data)
