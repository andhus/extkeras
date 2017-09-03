from __future__ import division, print_function

import tensorflow as tf
from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Recurrent, activations, initializers

from extkeras.layers.children_layers_mixin import ChildLayersMixin

"""Dynamic Associative Networks (DANs)

Yesterday in my (fewerish) sleep, I came up with a exciting idea. Probably
someone has already tried the same thing but I'm surprised I haven't heard more
about it. It should be trivial to implement so I'm just gonna go ahead and test
it.

In regular FFW NNets we feed an input through a predefined set stack of
transformations to get an output. We then compare the output with a target and
use backprop to adjust the weights of the stack of transformations.
I've always felt this is nothing evenly remotely close to intelligence or
"reasoning/problem solving".


With DANs the idea is to instead define a *dynamic (recurrent) system*. We use
the input to determine the initial state of the system then let the system "run
freely" until its state *converges*. We then use the converged state (or some
transformation from it) as output and compare it to the target. We then again
just use backprop to adjust the weights defining the dynamic system.


This way can can sort of "forget about depth" of the network - we let it run an
undefined number of transformations until it's "done" (i.e. converges).
A recurrent dynamic system should also be able to reuse "sub-modules/programs"
more several times during analysis of one input which should allow for more
efficient use of its weights. (This is a kind of echo-state network i suppose
I have to read up on those...).

The the question is how to implement this. I came up wih an idea that is
essentially a generalisation of MLP but where information does not only have to
flow in a single direction. Just like a DAG is a special case of a DG the MLP
would be a special case of a DANs.

Moreover my idea is to connect nodes in the DAN by "association keys" rather
explicit connections (surely someone must have tried this?). I.e. the strength
of the connection between node A -> B depends on how similar A:s outgoing
associative key is with B:s incoming associative key is. I find this setup
interesting for a number of reasons:
    1. it somehow enforces sparsity in the connections (more on this later)
    2. it makes the number of weights as well as the computational time for
        evaluating a single time step scale only *linearly* with the number of
        nodes in stead of quadratically in a regular fully connected networks

I just had to get this down into writing before starting to implement...

Plan:
(-) implement basic DAN layer by using keras Recurrent layer... (hide in single
    layer later)
(-) write basic tests of step function
(-) use loss weights that gradually promotes convergence
(-) train on minimal mock problem
(-) train on mnist
(-) ...
"""


class DANRecurrent(ChildLayersMixin, Recurrent):
    """Fully-connected RNN where the output is to be fed back to input.

    # Arguments TODO OLD from Simple Recurrent

        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.

    # References
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    def __init__(
        self,
        units,
        nodes,
        key_size,
        activation='tanh',
        keys_initializer='glorot_uniform',
        **kwargs
    ):
        super(DANRecurrent, self).__init__(**kwargs)
        self.units = units
        self.nodes = nodes
        self.key_size = key_size
        self.activation = activations.get(activation)

        self.keys_initializer = initializers.get(keys_initializer)
        self.state_spec = [
            InputSpec(shape=(None, self.units)),  # from values
            InputSpec(shape=(None, self.key_size))  # out_key for next step...
        ]

        assert not self.stateful

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        self.input_dim = input_shape[2]
        self.input_spec[0] = InputSpec(shape=(None, None, self.input_dim))

        self.states = [None, None]

        # self.input_kernel = self.add_weight(
        #     shape=(self.input_dim, self.nodes),
        #     name='input_kernel',
        #     initializer=self.keys_initializer,
        # )
        self.in_keys = self.add_weight(
            shape=(self.nodes, self.key_size),
            name='in_keys',
            initializer=self.keys_initializer,
        )
        self.out_keys = self.add_weight(
            shape=(self.nodes, self.key_size),
            name='out_keys',
            initializer=self.keys_initializer,
        )
        self.values = self.add_weight(
            shape=(self.nodes, self.units),
            name='values',
            initializer=self.keys_initializer,
        )
        self.built = True

    def preprocess_input(self, inputs, training=None):
        return inputs

    def step(self, inputs, states):
        if not self.implementation == 0:
            raise ValueError('')

        [in_key, value_tm1] = states[:2]

        out_key, value = self._step(in_key, value_tm1)

        return value, [out_key, value]

    def _step(self, in_key, value_tm1):
        edge_weights = K.sum(
            (
                K.expand_dims(in_key, 1) -
                K.expand_dims(K.softmax(self.in_keys), 0)
            ) ** 2,
            axis=-1,
            keepdims=True
        )
        edge_weights = tf.nn.softmax(edge_weights, dim=1)  # TODO no dim in keras softmax!?
        out_key = K.sum(
            edge_weights * K.softmax(self.out_keys),
            axis=1,
        )
        new_value = K.sum(
            edge_weights * K.softmax(self.values),
            axis=1,
        )
        value = value_tm1 * 0.9 + new_value * 0.1

        return out_key, value

    def get_constants(self, inputs, training=None):
        constants = []
        if self.implementation != 0 and 0 < self.dropout < 1:
            input_shape = K.int_shape(inputs)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = K.in_train_phase(dropped_inputs,
                                       ones,
                                       training=training)
            constants.append(dp_mask)
        else:
            constants.append(K.cast_to_floatx(1.))

        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)
            rec_dp_mask = K.in_train_phase(dropped_inputs,
                                           ones,
                                           training=training)
            constants.append(rec_dp_mask)
        else:
            constants.append(K.cast_to_floatx(1.))
        return constants

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'recurrent_dropout': self.recurrent_dropout
        }
        base_config = super(DANRecurrent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
