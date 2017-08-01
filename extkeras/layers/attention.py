from __future__ import division, print_function


from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Dense, concatenate
from keras.layers.recurrent import Recurrent
from keras.layers.recurrent import SimpleRNN as _SimpleRNN

from extkeras.layers.children_layers_mixin import ChildrenLayersMixin


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
        wrapped_recurrent_input = self.attention_layer.attention_step(
            attended=attended,
            recurrent_input=inputs,
            recurrent_states=list(states[:-2]),
            attention_states=[]  # TODO fix!
        )
        return self.recurrent_layer.step(wrapped_recurrent_input, states)

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

        :return: recurrent input, passed as inputs to RNN
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


class DenseStatelessAttention(AttentionLayer, ChildrenLayersMixin):

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
        return concatenate([recurrent_input, attention])
