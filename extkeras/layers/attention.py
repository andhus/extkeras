from __future__ import division, print_function

import abc

from collections import OrderedDict
from warnings import warn

import numpy as np

from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Dense, concatenate, Layer
from keras.layers.recurrent import Recurrent

from extkeras.layers.children_layers_mixin import ChildrenLayersMixin
from extkeras.layers.distribution import (
    MixtureDistributionABC,
    DistributionOutputLayer
)


class RecurrentAttentionWrapper(ChildrenLayersMixin, Recurrent):

    def __init__(
        self,
        recurrent_layer,
        return_attention=False,
        concatenate_input=True,
        attend_after=False,
        **kwargs
    ):
        super(RecurrentAttentionWrapper, self).__init__(**kwargs)
        self.recurrent_layer = self.add_child(
            'recurrent_layer',
            recurrent_layer
        )
        self.return_attention = return_attention
        self.concatenate_input = concatenate_input
        self.attend_after = attend_after

        self.input_spec = [InputSpec(ndim=3), None]

        self._attended_spec = InputSpec(ndim=2)
        self._attention_step_output_spec = InputSpec(ndim=2)
        self._attention_state_spec = [InputSpec(ndim=2)]
        self._attention_states = [None]

        # will be set in call, then passed to step by get_constants
        self._attended = None

    def _validate_wrapped_recurrent(self, recurrent_layer):
        wrapped_recurrent_expected_attrs = dict(
            return_sequences=False,
            return_state=False,
            go_backwards=False,
            stateful=False
        )
        for attr, expected_value in wrapped_recurrent_expected_attrs.items():
            if not getattr(recurrent_layer, attr) == expected_value:
                warn(
                    'non default value for {recurrent_class}.{attr}, '
                    'found {existing}, expected {expected}. This attribute '
                    'will be overwritten with expected value. The '
                    'corresponding argument should instead be passed to the '
                    '{self_class}'.format(
                        recurrent_class=recurrent_layer.__class__.__name__,
                        attr=attr,
                        existing=getattr(recurrent_layer, attr),
                        expected=expected_value,
                        self_class=self.__class__.__name__
                    )
                )
                setattr(recurrent_layer, attr, expected_value)

    @property
    def attended_spec(self):
        return self._attended_spec

    @property
    def attention_step_output_spec(self):
        return self._attention_step_output_spec

    @property
    def attention_state_spec(self):
        return self._attention_state_spec

    @property
    def attention_states(self):
        return self._attention_states

    @property
    def n_attention_states(self):
        return len(self.attention_states)

    @property
    def n_recurrent_states(self):
        return len(self.recurrent_layer.states)

    @abc.abstractmethod
    def attention_build(
        self,
        attended_shape,
        wrapped_step_input_shape,
        wrapped_state_shapes
    ):
        """Build transformations related to attention mechanism, will be called
        in build"""
        pass

    @abc.abstractproperty
    def attention_output_dim(self):
        """Must be specified independently  of input shape.

        Normally we would compute output shape by passing input shape but
        this would lead to infinite recursion (TODO explain).

        # Returns
            dimension of attention output (int)
        """
        pass

    @abc.abstractmethod
    def attention_step(
        self,
        attended,
        attention_states,
        step_input,
        recurrent_states,
    ):
        """

        # Args
            attended: the same tensor each timestep
            attention_states: states from previous attention step, by
                default attention from last step but can be extended
            step_input: the inputs at current timesteps
            recurrent_states: states for recurrent layer (excluding constants
                like dropout tensors) from previous state if attend_after=False
                otherwise from current time step.

        # Returns
            attention_h: the computed attention at current time step
            attention_states: states to be passed to next attention_step,
                normally this is just [attention_h], NOTE if more states are
                used these should be _appeded_ to attention states,
                attention_states[0] should always be attention_h.
        """
        pass

    def get_attention_initial_state(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.attention_output_dim])  # (samples, output_dim)
        return [initial_state]

    @property
    def states(self):
        return self.attention_states + self.recurrent_layer.states

    @property
    def state_specs(self):
        recurrent_state_spec = self.recurrent_layer.state_spec
        if not isinstance(recurrent_state_spec, list):
            recurrent_state_spec = [recurrent_state_spec]

        return self.attention_state_spec + recurrent_state_spec

    def compute_output_shape(self, input_shape):
        """"""
        [input_shape, attention_shape] = input_shape

        recurrent_output_shape = self._compute_recurrent_step_output_shape(
            input_shape
        )

        if self.return_sequences:
            if self.return_attention:
                output_shape = [
                    (input_shape[0], input_shape[1], self.attention_output_dim),
                    input_shape[:2] + recurrent_output_shape[1:]
                ]
            else:
                output_shape = input_shape[:2] + recurrent_output_shape[1:]
        else:
            if self.return_attention:
                output_shape = [
                    (input_shape[0], self.attention_output_dim),
                    recurrent_output_shape
                ]
            else:
                output_shape = recurrent_output_shape

        if self.return_state:
            if not isinstance(output_shape, list):
                output_shape = [output_shape]
            attention_state_shape = [
                (input_shape[0], spec.shape[-1])
                for spec in self.attention_state_spec
            ]
            recurrent_state_shape = [
                (input_shape[0], self.recurrent_layer.units)
                for _ in self.recurrent_layer.states
            ]
            return output_shape + attention_state_shape + recurrent_state_shape
        else:
            return output_shape

    def _compute_recurrent_step_output_shape(
        self,
        recurrent_input_shape,
    ):
        """Computes wrapped recurrent "step" output shape (no time/sequence
        dimension).

        Normally this output shape is simply:
            (input_shape[0], wrapped_recurrent.units)
        However the approach in this method is more safe for custom recurrent
        layers where this might not be the case.

        # Returns
            The wrapped recurrent (step) output shape (int, int)
        """
        wrapped_recurrent_input_shape = (
            recurrent_input_shape[0],
            recurrent_input_shape[1],
            self.attention_output_dim + recurrent_input_shape[-1]
            if self.concatenate_input else self.attention_output_dim
        )

        return self.recurrent_layer.compute_output_shape(
            wrapped_recurrent_input_shape
        )
        # it is verified that this will return the step output shape
        # since return sequences must be False in recurrent layer

    def build(self, input_shape):

        [input_shape, attended_shape] = input_shape

        self.input_spec = [  # TODO remove?
            InputSpec(shape=input_shape),
            InputSpec(shape=attended_shape)
        ]

        wrapped_step_input_shape = (input_shape[0], input_shape[-1])

        # TODO for existing keras recurrent layers state size is always units
        # but that is not very general...
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
        self._attended_spec = InputSpec(shape=attended_shape)

        wrapped_recurrent_input_shape = (
            input_shape[0],
            input_shape[1],
            self.attention_output_dim + input_shape[-1]
            if self.concatenate_input else self.attention_output_dim
        )

        self.recurrent_layer.build(wrapped_recurrent_input_shape)

        self.built = True

    def __call__(self, inputs, initial_state=None, **kwargs):
        outputs = super(RecurrentAttentionWrapper, self).__call__(
            inputs,
            initial_state=initial_state,
            **kwargs
        )
        if self.return_attention:
            output = outputs[0][:self.recurrent_layer.units]
            attention = output[0][self.recurrent_layer.units:]
            outputs = [output, attention] + outputs[1:]

        return outputs

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
        state_components = self.get_states_components(states)
        if self.attend_after:
            return self.attend_after_step(inputs, *state_components)
        else:
            return self.attend_before_step(inputs, *state_components)

    def attend_before_step(
        self,
        inputs,
        attended,
        attention_states_tm1,
        recurrent_states_tm1,
        recurrent_constants
    ):
        attention_h, attention_states = \
            self.attention_step(
                attended=attended,
                attention_states=attention_states_tm1,
                step_input=inputs,
                recurrent_states=recurrent_states_tm1,
            )
        if self.concatenate_input:
            recurrent_input = concatenate([attention_h, inputs])
        else:
            recurrent_input = attention_h

        output, recurrent_states = self.recurrent_layer.step(
            recurrent_input,
            recurrent_states_tm1 + recurrent_constants
        )

        if self.return_attention:
            output = concatenate([output, attention_h])

        return output, attention_states + recurrent_states

    def attend_after_step(
        self,
        inputs,
        attended,
        attention_states_tm1,
        recurrent_states_tm1,
        recurrent_constants
    ):
        attention_h_tm1 = attention_states_tm1[0]

        if self.concatenate_input:
            recurrent_input = concatenate([attention_h_tm1, inputs])
        else:
            recurrent_input = attention_h_tm1

        output, recurrent_states = self.recurrent_layer.step(
            recurrent_input,
            recurrent_states_tm1 + recurrent_constants
        )

        attention_h, attention_states = \
            self.attention_step(
                attended=attended,
                attention_states=attention_states_tm1,
                step_input=inputs,
                recurrent_states=recurrent_states
            )

        if self.return_attention:
            output = concatenate([output, attention_h])

        return output, attention_states + recurrent_states

    def get_states_components(self, states):
        states = list(states)  # is tuple before?
        states_, attended = states[:-1], states[-1]
        attention_states = states_[:self.n_attention_states]
        states__ = states[self.n_attention_states:]
        recurrent_states = states__[:self.n_recurrent_states]
        recurrent_constants = states__[self.n_recurrent_states:]
        # TODO add support for attention constants?
        return (
            attended,
            attention_states,
            recurrent_states,
            recurrent_constants
        )

    def get_initial_state(self, inputs):
        return (
            self.get_attention_initial_state(inputs) +
            self.recurrent_layer.get_initial_state(inputs)
        )

    def preprocess_input(self, inputs, training=None):
        # TODO disable!?
        return self.recurrent_layer.preprocess_input(inputs, training=training)


class DenseStatelessAttention(RecurrentAttentionWrapper):

    def __init__(self, units, **kwargs):
        super(DenseStatelessAttention, self).__init__(**kwargs)
        self.units = units

    def attention_build(
        self,
        attended_shape,
        wrapped_step_input_shape,
        wrapped_state_shapes
    ):
        """Build transformations related to attention mechanism, will be called
        in build"""
        self.dense = self.add_child('dense', Dense(self.units))
        self.dense.build(
            (
                attended_shape[0],
                (
                    attended_shape[1] +
                    wrapped_step_input_shape[1] +
                    sum([s[1] for s in wrapped_state_shapes])
                )
            )
        )

    @property
    def attention_output_dim(self):
        return self.units

    def attention_step(
        self,
        attended,
        attention_states,
        step_input,
        recurrent_states,
    ):
        attention_h = self.dense(
            concatenate([attended, step_input] + recurrent_states)
        )
        return attention_h, [attention_h]


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


class AlexGravesSequenceAttention(RecurrentAttentionWrapper):

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
        self._attention_states = [None, None]
        self._attention_state_spec = [
            InputSpec(ndim=2),          # attention (tm1)
            InputSpec(shape=(None, 1))  # kappa
        ]

    def attention_build(
        self,
        attended_shape,
        wrapped_step_input_shape,
        wrapped_state_shapes
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
            wrapped_step_input_shape[-1] + wrapped_state_shapes[0]
        )
        self.params_layer.build(input_shape)

    def get_attention_initial_state(self, inputs):
        [attention_tm1_state] = super(
            AlexGravesSequenceAttention,
            self
        ).get_attention_initial_state(inputs)
        kappa_tm1 = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        kappa_tm1 = K.sum(kappa_tm1, axis=(1, 2))  # (samples,)
        kappa_tm1 = K.expand_dims(kappa_tm1)  # (samples, 1)
        kappa_tm1 = K.tile(kappa_tm1, [1, self.distribution.n_components])  # (samples, n_components)

        return [attention_tm1_state, kappa_tm1]

    def attention_step(
        self,
        attended,
        attention_states,
        step_input,
        recurrent_states
    ):
        [attention_tm1, kappa_tm1] = attention_states
        params = self.params_layer(
            concatenate([step_input, recurrent_states[0]])
        )
        attention, kappa = self._get_attention_and_kappa(
            attended,
            params,
            kappa_tm1
        )
        return attention, [attention, kappa]

    def _get_attention_and_kappa(self, attended, params, kappa_tm1):
        """
        # Args
            params: the params of this distribution
            attended: the attended sequence (samples, timesteps, features)
        # Returns
            attention tensor (samples, features)
        """
        att_idx = K.constant(np.arange(self.attended_shape[1])[None, :, None])
        alpha, beta, kappa_diff = self.distribution.split_param_types(params)
        kappa = kappa_diff + kappa_tm1

        kappa_ = K.expand_dims(kappa, 1)
        beta_ = K.expand_dims(beta, 1)
        alpha_ = K.expand_dims(alpha, 1)

        attention_w = K.sum(
            alpha_ * K.exp(- beta_ * K.square(kappa_ - att_idx)),
            axis=-1,
            # keepdims=True
        )
        attention_w = K.expand_dims(attention_w, -1)  # TODO remove and keepdims
        attention = K.sum(attention_w * attended, axis=1)

        return attention, kappa[:, 0, :]
