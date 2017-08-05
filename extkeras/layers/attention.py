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
from extkeras.layers.distribution import MixtureDistributionABC, \
    DistributionOutputLayer


class RecurrentAttentionWrapper(ChildrenLayersMixin, Recurrent):

    def __init__(
        self,
        recurrent_layer,
        return_attention=False,
        concatenate_input=True,  # TODO auto concat input to
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

        self._attention_input_spec = None
        self._attention_step_output_spec = None
        self._attention_state_spec = []
        self._attention_states = []
        # self._attention_fwd_state = None

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
    def attention_input_spec(self):
        return self._attention_input_spec

    @property
    def attention_step_output_spec(self):
        return self._attention_step_output_spec

    @property
    def attention_state_spec(self):
        return self._attention_state_spec

    # @property
    # def attention_state_and_fwd_spec(self):
    #     """
    #     Returns
            # list (possibly empty) of InputSpec
        # """
        # if self.attend_after:
        #     return self._attention_state_spec + [self.attention_step_output_spec]
        # else:
        #     return self._attention_state_spec

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
        """"""
        pass

    @abc.abstractmethod
    def compute_attention_step_output_shape(self, step_input_shape):
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

        :param attended: the same tensor each timestep
        :param attention_states: states from previous attention step
        :param step_input: the inputs at current timesteps
        :param recurrent_states: from previous state if attend before or from
            this timestep if attend after.
        :return:
        """
        pass

    @property
    def input_spec(self):
        return [self.attention_input_spec, self.recurrent_layer.input_spec]

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
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if self.return_sequences:
            output_shape = (
                input_shape[0],
                input_shape[1],
                self.recurrent_layer.units
            )
        else:
            output_shape = (input_shape[0], self.recurrent_layer.units)

        # TODO modify if return_attention !

        if self.return_state:
            # TODO must be fixed wrt attention states
            state_shape = [(input_shape[0], self.units) for _ in self.states]
            return [output_shape] + state_shape
        else:
            return output_shape

    # def compute_output_shape(self, input_shape):
    #     """"""
    #     if isinstance(input_shape, list):
    #         [input_shape] = input_shape
    #
    #     recurrent_output_shape = self._compute_wrapped_recurrent_output_shape(
    #         input_shape
    #     )
    #
    #     if self.return_sequences:
    #         output_shape = input_shape[:2] + recurrent_output_shape[1:]
    #     else:
    #         output_shape = recurrent_output_shape
    #
    #     if self.return_state:
    #         pass
    #         # TODO
    #     #     state_shape = [(input_shape[0], self.units) for _ in self.states]
    #     #     return [output_shape] + state_shape
    #     # else:
    #     #     return output_shape
    #
    # def _compute_wrapped_recurrent_output_shape(
    #     self,
    #     input_shape,
    # ):
    #     """Computes wrapped recurrent "step" output shape (no time/sequence
    #     dimension).
    #
    #     Normally this output shape should be independent if attention is
    #     applied before or after and normally it is simply:
    #         (input_shape[0], wrapped_recurrent.units)
    #     However the approach in this method is more safe for custom recurrent
    #     layers where this might not be the case.
    #
    #     # Returns
    #         The wrapped recurrent (step) output shape (int, int)
    #     """
    #     [attended_shape, input_shape] = input_shape
    #     if self.attend_after:
    #         wrapped_recurrent_input_shape = input_shape
    #     else:  # attend before (default)
    #         step_input_shape = (input_shape[0], input_shape[-1])
    #         attention_step_output_shape = self.compute_attention_step_output_shape(
    #             step_input_shape
    #         )
    #         wrapped_recurrent_input_shape = (
    #             input_shape[0],
    #             input_shape[1],
    #             attention_step_output_shape[-1] + input_shape[-1]
    #             if self.concatenate_input else attention_step_output_shape[-1]
    #         )
    #
    #     return self.recurrent_layer.compute_output_shape(
    #         wrapped_recurrent_input_shape
    #     )

    def build(self, input_shape):
        [attended_shape, input_shape] = input_shape

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
        if self.attend_after:
            return self.attend_before_step(inputs, states)
        else:
            return self.attend_after_step(inputs, states)

    def attend_before_step(self, inputs, states):
        attended, attention_states, recurrent_states, recurrent_constants = \
            self.get_states_components(states)

        attention_h, attention_states = \
            self.attention_step(
                attended=attended,
                attention_states=attention_states,
                recurrent_input=inputs,
                recurrent_states=recurrent_states,
            )
        if self.concatenate_input:
            recurrent_input = concatenate([attention_h, inputs])
        else:
            recurrent_input = attention_h

        output, recurrent_states_and_constants = self.recurrent_layer.step(
            recurrent_input,
            recurrent_states + recurrent_constants
        )

        if self.return_attention:
            output = concatenate(output, attention_h)

        return output, attention_states + recurrent_states_and_constants

    def attend_after_step(self, inputs, states):
        attended, attention_states_tm1, recurrent_states_tm1, recurrent_constants = \
            self.get_states_components(states)

        attention_states_tm1, attention_h_tm1 = attention_states_tm1[:-1], attention_states_tm1[-1]
        # TODO the above not yet supported!

        if self.concatenate_input:
            recurrent_input = concatenate(attention_h_tm1, inputs)
        else:
            recurrent_input = attention_h_tm1

        output, recurrent_states_and_constants = self.recurrent_layer.step(
            recurrent_input,
            recurrent_states_tm1 + recurrent_constants
        )

        attention_h, attention_states = \
            self.attention_step(
                attended=attended,
                attention_states=attention_states_tm1,
                step_input=inputs,
                recurrent_states=(
                    recurrent_states_and_constants[:self.n_recurrent_states]
                )
            )

        attention_states.append(attention_h)

        if self.return_attention:
            output = concatenate(output, attention_h)

        return output, attention_states + recurrent_states_and_constants

    def get_states_components(self, states):
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
