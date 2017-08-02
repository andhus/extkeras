from __future__ import division, print_function

from collections import OrderedDict

import numpy as np

from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Dense, concatenate, Layer
from keras.layers.recurrent import Recurrent

from extkeras.layers.children_layers_mixin import ChildrenLayersMixin
from extkeras.layers.distribution import MixtureDistributionABC, \
    DistributionOutputLayer


class RecurrentAttentionWrapper(ChildrenLayersMixin, Recurrent):

    def __init__(self, attention_layer, recurrent_layer):
        super(RecurrentAttentionWrapper, self).__init__()
        self.attention_layer = self.add_child(
            'attention_layer',
            attention_layer
        )
        self.recurrent_layer = self.add_child(
            'recurrent_layer',
            recurrent_layer
        )
        self._attended = None
        self.input_spec = [InputSpec(ndim=3), None]
        # later should be set to attention_layer.input_spec

    @property
    def states(self):
        return self.recurrent_layer.states + self.attention_layer.states

    def compute_output_shape(self, input_shape):
        [input_shape, attended_shape] = input_shape
        input_step_shape = input_shape[:1] + input_shape[-1:]
        wrapped_recurrent_input_shape = (
            input_shape[:-1] +
            self.attention_layer.compute_output_shape(
                input_step_shape
            )[-1:]
        )

        return self.recurrent_layer.compute_output_shape(
            wrapped_recurrent_input_shape
        )

    def build(self, input_shape):
        [input_shape, attended_shape] = input_shape
        wrapped_recurrent_step_input_shape = input_shape[:1] + input_shape[-1:]

        wrapped_recurrent_state_shapes = [
            input_shape[:1] + spec.shape[1:]
            for spec in self.recurrent_layer.state_spec
        ] if isinstance(self.recurrent_layer.state_spec, list) else [(
            input_shape[:1] + self.recurrent_layer.state_spec.shape[1:]
        )]

        self.attention_layer.build(
            attended_shape,
            wrapped_recurrent_step_input_shape,
            wrapped_recurrent_state_shapes
        )
        wrapped_recurrent_input_shape = (
            input_shape[:-1] +
            self.attention_layer.compute_output_shape(
                wrapped_recurrent_step_input_shape
            )[-1:]
        )
        self.recurrent_layer.build(wrapped_recurrent_input_shape)

        self.input_spec = [InputSpec(ndim=3), InputSpec(shape=attended_shape)]
        self.built = True

    def call(
        self,
        inputs,
        mask=None,
        training=None,
        initial_state=None,
    ):
        inputs, self._attended = inputs
        return super(RecurrentAttentionWrapper, self).call(
            inputs,
            mask=mask,
            training=training,
            initial_state=initial_state,
        )

    def get_constants(self, inputs, training=None):
        constants = self.recurrent_layer.get_constants(
            inputs,
            training=training
        )
        constants.append(self._attended)

        return constants

    def step(self, inputs, states):
        attended = states[-1]
        states = states[:-1]
        wrapped_recurrent_input, attention_states = \
            self.attention_layer.attention_step(
                attended=attended,
                recurrent_input=inputs,
                recurrent_states=list(states[:-2]),
                attention_states=[]  # TODO fix!
            )
        output, wrapped_states = self.recurrent_layer.step(
            wrapped_recurrent_input,
            states  # TODO remove attention states
        )
        new_states = wrapped_states + attention_states

        return output, new_states

    # simply bypassed methods
    def get_initial_state(self, inputs):
        return self.recurrent_layer.get_initial_state(inputs) + []  # TODO support attention states

    def preprocess_input(self, inputs, training=None):
        return self.recurrent_layer.preprocess_input(inputs, training=training)


class AttentionLayer(Recurrent):

    def __init__(
        self,
        units,
        *args,
        **kwargs
    ):
        super(AttentionLayer, self).__init__(*args, **kwargs)
        self.units = units

    @property
    def states(self):
        return []

    def attention_step(
        self,
        attended,
        recurrent_input,
        recurrent_states,
        attention_states
    ):
        """
        :param inputs:
        :param states:

        :return: (recurrent input, attention_states), first will be passed as
            inputs to RNN
        """
        raise NotImplementedError('')

    def compute_output_shape(self, input_shape):
        # default is concatenation of input and attention
        return (input_shape[0], input_shape[1] + self.units)

    def build(
        self,
        attended_shape,
        recurrent_step_input_shape,
        recurrent_state_shapes
    ):
        self.built = True

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            'RecurrentAttentionLayer should only be used within a '
            'RecurrentAttentionWrapper'
        )


class DenseStatelessAttention(ChildrenLayersMixin, AttentionLayer):

    def build(
        self,
        attended_shape,
        recurrent_step_input_shape,
        recurrent_state_shapes
    ):
        self.dense = self.add_child('dense', Dense(self.units))
        self.dense.build(
            (
                attended_shape[0],
                (
                    attended_shape[1] +
                    recurrent_step_input_shape[1] +
                    sum([s[1] for s in recurrent_state_shapes])
                )
            )
        )
        self.built = True

    def attention_step(
        self,
        attended,
        recurrent_input,
        recurrent_states,
        attention_states
    ):
        assert not attention_states
        attention = self.dense(
            concatenate([attended, recurrent_input] + recurrent_states)
        )
        return concatenate([recurrent_input, attention]), []


class AlexGravesSequenceAttentionParams(MixtureDistributionABC):
    """NON-NORMALISED 1D Mixture of gaussian distribution"""

    def __init__(
        self,
        n_components,
        alpha_activation=None,
        beta_activation=None,
        kappa_activation=None,
    ):
        super(AlexGravesSequenceAttentionParams, self).__init__(n_components)
        self.alpha_activation = alpha_activation or K.exp
        self.beta_activation = beta_activation or K.exp
        self.kappa_activation = kappa_activation or K.exp

    @property
    def param_type_to_size(self):
        return OrderedDict([
            ('alpha', self.n_components),
            ('beta', self.n_components),
            ('kappa', self.n_components)
        ])

    def activation(self, x):
        _alpha, _beta, _kappa = self.split_param_types(x)
        alpha = self.alpha_activation(_alpha)
        beta = self.beta_activation(_kappa)
        kappa = self.kappa_activation(_beta)

        return concatenate([alpha, beta, kappa], axis=-1)

    def loss(self, y_true, y_pred):
        raise NotImplementedError('')


class AlexGravesSequenceAttention(ChildrenLayersMixin, AttentionLayer):

    def __init__(
        self,
        n_components,
        alpha_activation=None,
        beta_activation=None,
        kappa_activation=None,
        *args,
        **kwargs
    ):
        super(AlexGravesSequenceAttention, self).__init__(*args, **kwargs)
        self.distribution = AlexGravesSequenceAttentionParams(
            n_components,
            alpha_activation,
            beta_activation,
            kappa_activation,
        )

    def build(
        self,
        attended_shape,
        recurrent_step_input_shape,
        recurrent_state_shapes
    ):
        self.attended_shape = attended_shape
        self.params_layer = self.add_child(
            'params_layer',
            DistributionOutputLayer(
                self.distribution
            )
        )
        input_shape = (
            attended_shape[0],
            recurrent_step_input_shape[-1] + recurrent_state_shapes[0]
        )
        self.params_layer.build(input_shape)

    def attention_step(
        self,
        attended,
        recurrent_input,
        recurrent_states,
        attention_states
    ):
        [kappa_tm1] = attention_states[0]
        params = self.params_layer(
            concatenate([recurrent_input, recurrent_states[0]])
        )
        attention, kappa = self.get_attention(params, attended, kappa_tm1)
        concatenate([recurrent_input, attention]), [kappa]

    def get_attention(self, params, attended, kappa_tm1):
        """
        # Args
            params: the params of this distribution
            attended: the attended sequence (samples, timesteps, features)
        # Returns
            attention tensor (samples, features)
        """
        att_idx = K.constant(np.arange(self.attended_shape[1])[None, :, None])
        alpha, beta, kappa = self.distribution.split_param_types(params)
        kappa = K.expand_dims(kappa + kappa_tm1, 1)
        beta = K.expand_dims(beta, 1)
        alpha = K.expand_dims(alpha, 1)
        attention_w = K.sum(
            alpha * K.exp(- beta * K.square(kappa - att_idx)),
            axis=-1
        )
        attention_w = K.expand_dims(attention_w, -1)
        attention = K.sum(attention_w * attended, axis=1)

        return attention, kappa[:, 0, :]
