from __future__ import division, print_function


from keras import backend as K
from keras.layers.recurrent import Recurrent
from keras.layers.recurrent import SimpleRNN as _SimpleRNN

from extkeras.layers.children_layers_mixin import ChildLayersMixin


class RecurrentAttentionWrapper(ChildLayersMixin, Recurrent):

    def __init__(self, attention_model, recurrent_layer):
        super(RecurrentAttentionWrapper, self).__init__()
        self.attention_model = self.add_child(
            'attention_model',
            attention_model
        )
        self.recurrent_layer = self.add_child(
            'recurrent_layer',
            recurrent_layer
        )
        self._attended = None

    @property
    def states(self):
        return self.recurrent_layer.states

    def compute_output_shape(self, input_shape):
        # TODO modify input shape!!
        return self.recurrent_layer.compute_output_shape(input_shape)

    def build(self, input_shape):
        self.recurrent_layer.build(input_shape)
        # self.attention_model.build(input_shape) ??

        # TODO
        # self.attention_model.build(input_shape[0])
        # TODO compute input shape for rnn based on output for attention...
        # self.recurrent_layer.build(input_shape[0])

    def call(
        self,
        inputs,
        mask=None,
        training=None,
        initial_state=None,
        attended=None
    ):
        self._attended = attended
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
        h_attention = self.attention_model(attended)
        inputs = inputs + h_attention  # temp hack to just sum
        return self.recurrent_layer.step(inputs, states)

    # simply bypassed methods
    def get_initial_state(self, inputs):
        return self.recurrent_layer.get_initial_state(inputs)

    def preprocess_input(self, inputs, training=None):
        return self.recurrent_layer.preprocess_input(inputs, training=training)


class RecurrentAttentionLayer(Recurrent):

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            'RecurrentAttentionLayer should only be used within a '
            'RecurrentAttentionWrapper'
        )

    def attention_step(
        self,
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
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1], self.units)
        else:
            output_shape = (input_shape[0], self.units)

        if self.return_state:
            state_shape = [(input_shape[0], self.units) for _ in self.states]
            return [output_shape] + state_shape
        else:
            return output_shape
