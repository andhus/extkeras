from __future__ import division, print_function

import abc

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

    def __init__(
        self,
        recurrent_layer,
        output_attention=False,
        concatenate_input=True,  # TODO auto concat input to
        attend_after=False
    ):
        super(RecurrentAttentionWrapper, self).__init__()
        self.recurrent_layer = self.add_child(
            'recurrent_layer',
            recurrent_layer
        )
        self.output_attention = output_attention
        self.concatenate_input = concatenate_input
        self.attend_after = attend_after
        self._attention_input_spec = None
        self._attended = None

    @property
    def attention_input_spec(self):
        return self._attention_input_spec

    @property
    def attention_states(self):
        return []

    @abc.abstractmethod
    def attention_build(
        self,
        attended_shape,
        wrapped_step_input_shape,
        wrapped_state_shapes
    ):
        pass

    @abc.abstractmethod
    def compute_attention_step_output_shape(self, step_input_shape):
        pass

    @abc.abstractmethod
    def attention_step(
        self,
        attended,
        recurrent_input,
        recurrent_states,
        attention_states
    ):
        pass

    @property
    def input_spec(self):
        return [self.recurrent_layer.input_spec, self.attention_input_spec]

    @property
    def states(self):
        return self.recurrent_layer.states + self.attention_states

    def compute_output_shape(self, input_shape):
        if self.attend_after:
            return self._compute_attend_after_output_shape(input_shape)
        else:
            return self._compute_attend_before_output_shape(input_shape)

    def _compute_attend_before_output_shape(self, input_shape):
        """Note output shape should be independent if applied before or after
        but since input shape to wrapped recurrent differs and should be passed
        to compute output shape of recurrent it is done separately here...

        NOTE return output shape(s) if return sequences....
        """
        [input_shape, attended_shape] = input_shape
        step_input_shape = input_shape[:1] + input_shape[-1:]
        attention_step_output_shape = self.compute_attention_step_output_shape(
            step_input_shape
        )
        if self.concatenate_input:
            wrapped_recurrent_input_shape = (
                input_shape[:-1] +
                [attention_step_output_shape[-1] + input_shape[-1]]  # must be tuple?
            )
        else:
            wrapped_recurrent_input_shape = (
                input_shape[:-1] + attention_step_output_shape[-1:]
            )

        recurrent_output_shape = self.recurrent_layer.compute_output_shape(
            wrapped_recurrent_input_shape
        )
        if self.output_attention:
            return [recurrent_output_shape, attention_step_output_shape]
        else:
            return recurrent_output_shape

    def _compute_attend_after_output_shape(self, input_shape):
        raise NotImplementedError('')

    def build(self, input_shape):
        [input_shape, attended_shape] = input_shape
        wrapped_step_input_shape = input_shape[:1] + input_shape[-1:]

        wrapped_state_shapes = [
            input_shape[:1] + spec.shape[1:]
            for spec in self.recurrent_layer.state_spec
        ] if isinstance(self.recurrent_layer.state_spec, list) else [(
            input_shape[:1] + self.recurrent_layer.state_spec.shape[1:]
        )]

        self.attention_build(
            attended_shape,
            wrapped_step_input_shape,
            wrapped_state_shapes
        )
        wrapped_input_shape = (
            input_shape[:-1] +
            self.compute_attention_step_output_shape(
                wrapped_step_input_shape
            )[-1:]
        )
        self.recurrent_layer.build(wrapped_input_shape)

        self._attention_input_spec = InputSpec(shape=attended_shape)
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
        if self.attend_after:
            return self.attend_before_step(inputs, states)
        else:
            return self.attend_after_step(inputs, states)

    def attend_before_step(self, inputs, states):
        attended = states[-1]
        states = states[:-1]
        wrapped_recurrent_input, attention_states = \
            self.attention_step(
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

    def attend_before_step(self, inputs, states):
        raise NotImplementedError('')

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
