from __future__ import print_function, division

from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.engine.base_layer import _collect_previous_mask
from keras.layers import Layer, InputSpec
from keras.layers import concatenate
from keras.utils.generic_utils import has_arg


class AttentionCellWrapper(Layer):
    """Base class for recurrent attention mechanisms.

    This base class implements the RNN cell interface and defines a standard
    way for attention mechanisms to interact with a wrapped RNNCell
    (such as the `SimpleRNNCell`, `GRUCell` or `LSTMCell`).

    The main idea is that the attention mechanism, implemented by
    `attention_call` in extensions of this class, computes an "attention
    encoding", based on the attended input as well as the input and the wrapped
    cell state(s) at the current time step, which will be used as modified
    input for the wrapped cell.

    # Arguments
        cell: A RNN cell instance. The cell to wrap by the attention mechanism.
            See docs of `cell` argument in the `RNN` Layer for further details.
        attend_after: Boolean (default False). If True, the attention
            transformation defined by `attention_call` will be applied after
            the wrapped cell transformation (and the attention encoding will be
            used as input for wrapped cell transformation next time step).
        input_mode: String, one of `"replace"` (default) or `"concatenate"`.
            `"replace"`: only the attention encoding will be used as input for
                the wrapped cell.
            `"concatenate"` the concatenation of the original input and the
                attention encoding will be used as input to the wrapped cell.
            TODO set "concatenate" to default?
        output_mode: String, one of `"cell_output"` (default) or `"concatenate"`.
            `"cell_output"`: the output from the wrapped cell will be used.
            `"concatenate"`: the attention encoding will be concatenated to the
                output of the wrapped cell.
            TODO set "concatenate" to default?

    # Abstract Methods and Properties
        Extension of this class must implement:
            - `attention_build` (method): Builds the attention transformation
              based on input shapes.
            - `attention_call` (method): Defines the attention transformation
              returning the attention encoding.
            - `attention_size` (property): After `attention_build` has been
              called, this property should return the size (int) of the
              attention encoding. Do this by setting `_attention_size` in scope
              of `attention_build` or by implementing `attention_size`
              property.
        Extension of this class can optionally implement:
            - `attention_state_size` (property): Default [`attention_size`].
              If the attention mechanism has it own internal states (besides
              the attention encoding which is by default the only part of
              `attention_states`) override this property accordingly.
        See docs of the respective method/property for further details.

    # Details of interaction between attention and cell transformations
        Let "cell" denote wrapped RNN cell and "att(cell)" the complete
        attentive RNN cell defined by this class. We write the wrapped cell
        transformation as:

            y{t}, s_cell{t+1} = cell.call(x{t}, s_cell{t})

        where y{t} denotes the output, x{t} the input at and s_cell{t} the wrapped
        cell state(s) at time t and s_cell{t+1} the updated state(s).

        We can then write the complete "attentive" cell transformation as:

            y{t}, s_att(cell){t+1} = att(cell).call(x{t}, s_att(cell){t},
                                                    constants=attended)

        where s_att(cell) denotes the complete states of the attentive cell,
        which consists of the wrapped cell state(s) followed but the attention
        state(s), and attended denotes the tensor attended to (note: no time
        indexing as this is the same constant input at each time step).

        Internally, this is how the attention transformation, implemented by
        `attention_call`, interacts with the wrapped cell transformation
        `cell.call`:

        - with `attend_after=False` (default):
            a{t}, s_att{t+1} = att(cell).attention_call(x_t, s_cell{t},
                                                        attended, s_att{t})
            with `input_mode="replace"` (default):
                x'{t} = a{t}
            with `input_mode="concatenate"`:
                x'{t} = [x{t}, a{t}]

            y{t}, s_cell{t+1} = cell.call(x'{t}, s_cell{t})

        - with `attend_after=True`:
            with `input_mode="replace"` (default):
                x'{t} = a{t}
            with `input_mode="concatenate"`:
                x'{t} = [x{t}, a{t}]

            y{t}, s_cell{t+1} = cell.call(x'{t}, s_cell{t})
            a{t}, s_att{t+1} = att(cell).attention_call(x_t, s_cell{t+1},
                                                        attended, s_att{t})

        where a{t} denotes the attention encoding, s_att{t} the attention
        state(s), x'{t} the modified wrapped cell input and [x{.}, a{.}] the
        (tensor) concatenation of the input and attention encoding.
    """
    # in/output modes
    _REPLACE = "replace"
    _CELL_OUTPUT = "cell_output"
    _CONCATENATE = "concatenate"
    _input_modes = [_REPLACE, _CONCATENATE]
    _output_modes = [_CELL_OUTPUT, _CONCATENATE]

    def __init__(self, cell,
                 attend_after=False,
                 input_mode="replace",
                 output_mode="cell_output",
                 **kwargs):
        self.cell = cell  # must be set before calling super
        super(AttentionCellWrapper, self).__init__(**kwargs)
        self.attend_after = attend_after
        if input_mode not in self._input_modes:
            raise ValueError(
                "input_mode must be one of {}".format(self._input_modes))
        self.input_mode = input_mode
        if output_mode not in self._output_modes:
            raise ValueError(
                "output_mode must be one of {}".format(self._output_modes))
        self.output_mode = output_mode
        self.attended_spec = None
        self._attention_size = None

    def attention_call(self,
                       inputs,
                       cell_states,
                       attended,
                       attention_states,
                       attended_mask,
                       training=None):
        """The main logic for computing the attention encoding.

        # Arguments
            inputs: The input at current time step.
            cell_states: States for the wrapped RNN cell.
            attended: The constant tensor(s) to attend at each time step.
            attention_states: States dedicated for the attention mechanism.
            attended_mask: Collected masks for the attended.
            training: Whether run in training mode or not.

        # Returns
            attention_h: The computed attention encoding at current time step.
            attention_states: States to be passed to next `attention_call`. By
                default this should be [`attention_h`].
                NOTE: if additional states are used, these should be appended
                after `attention_h`, i.e. `attention_states[0]` should always
                be `attention_h`.
        """
        raise NotImplementedError(
            '`attention_call` must be implemented by extensions of `{}`'.format(
                self.__class__.__name__))

    def attention_build(self, input_shape, cell_state_size, attended_shape):
        """Build the attention mechanism.

        NOTE: `self._attention_size` should be set in this method to the size
        of the attention encoding (i.e. size of first `attention_states`)
        unless `attention_size` property is implemented in another way.

        # Arguments
            input_shape: Tuple of integers. Shape of the input at a single time
                step.
            cell_state_size: List of tuple of integers.
            attended_shape: List of tuple of integers.

            NOTE: both `cell_state_size` and `attended_shape` will always be
            lists - for simplicity. For example: even if (wrapped)
            `cell.state_size` is an integer, `cell_state_size` will be a list
            of this one element.
        """
        raise NotImplementedError(
            '`attention_build` must be implemented by extensions of `{}`'.format(
                self.__class__.__name__))

    @property
    def attention_size(self):
        """Size off attention encoding, an integer.
        """
        if self._attention_size is None and self.built:
            raise NotImplementedError(
                'extensions of `{}` must either set property `_attention_size`'
                ' in `attention_build` or implement the or implement'
                ' `attention_size` in some other way'.format(
                    self.__class__.__name__))

        return self._attention_size

    @property
    def attention_state_size(self):
        """Size of attention states, defaults to `attention_size`, an integer.

        Modify this property to return list of integers if the attention
        mechanism has several internal states. Note that the first size should
        always be the size of the attention encoding, i.e.:
            `attention_state_size[0]` = `attention_size`
        """
        return self.attention_size

    @property
    def state_size(self):
        """Size of states of the complete attentive cell, a tuple of integers.

        The attentive cell's states consists of the wrapped RNN cell state size(s)
        followed by attention state size(s). NOTE it is important that the wrapped
        cell states are first as the first state of any RNN cell should be same
        as the cell's output.
        """
        state_size_s = []
        for state_size in [self.cell.state_size, self.attention_state_size]:
            if hasattr(state_size, '__len__'):
                state_size_s += list(state_size)
            else:
                state_size_s.append(state_size)

        return tuple(state_size_s)

    @property
    def output_size(self):
        if self.output_mode == self._CELL_OUTPUT:
            return self._wrapped_cell_output_size
        if self.output_mode == self._CONCATENATE:
            return self._wrapped_cell_output_size + self.attention_size
        raise RuntimeError(  # already validated in __init__
            "got unexpected output_mode: {}".format(self.output_mode))

    def call(self, inputs, states, constants, training=None):
        """Complete attentive cell transformation.
        """
        attended = constants
        attended_mask = _collect_previous_mask(attended)
        # attended and mask are always lists for uniformity:
        if not isinstance(attended_mask, list):
            attended_mask = [attended_mask]
        cell_states = states[:self._num_wrapped_states]
        attention_states = states[self._num_wrapped_states:]

        if self.attend_after:
            call = self._call_attend_after
        else:
            call = self._call_attend_before

        return call(inputs=inputs,
                    cell_states=cell_states,
                    attended=attended,
                    attention_states=attention_states,
                    attended_mask=attended_mask,
                    training=training)

    def _call_attend_before(self,
                            inputs,
                            cell_states,
                            attended,
                            attention_states,
                            attended_mask,
                            training=None):
        """Complete attentive cell transformation, if `attend_after=False`.
        """
        attention_h, new_attention_states = self.attention_call(
            inputs=inputs,
            cell_states=cell_states,
            attended=attended,
            attention_states=attention_states,
            attended_mask=attended_mask,
            training=training)

        cell_input = self._get_cell_input(inputs, attention_h)

        if has_arg(self.cell.call, 'training'):
            cell_output, new_cell_states = self.cell.call(
                cell_input, cell_states, training=training)
        else:
            cell_output, new_cell_states = self.cell.call(cell_input, cell_states)

        output = self._get_output(cell_output, attention_h)

        return output, new_cell_states + new_attention_states

    def _call_attend_after(self,
                           inputs,
                           cell_states,
                           attended,
                           attention_states,
                           attended_mask,
                           training=None):
        """Complete attentive cell transformation, if `attend_after=True`.
        """
        attention_h_previous = attention_states[0]

        cell_input = self._get_cell_input(inputs, attention_h_previous)

        if has_arg(self.cell.call, 'training'):
            cell_output, new_cell_states = self.cell.call(
                cell_input, cell_states, training=training)
        else:
            cell_output, new_cell_states = self.cell.call(cell_input, cell_states)

        attention_h, new_attention_states = self.attention_call(
            inputs=inputs,
            cell_states=new_cell_states,
            attended=attended,
            attention_states=attention_states,
            attended_mask=attended_mask,
            training=training)

        output = self._get_output(cell_output, attention_h)

        return output, new_cell_states, new_attention_states

    def _get_cell_input(self, inputs, attention_h):
        if self.input_mode == self._REPLACE:
            return attention_h
        if self.input_mode == self._CONCATENATE:
            return concatenate([inputs, attention_h])
        raise RuntimeError(  # already validated in __init__
            "got unexpected input_mode: {}".format(self.input_mode))

    def _get_output(self, cell_output, attention_h):
        if self.output_mode == self._CELL_OUTPUT:
            return cell_output
        if self.output_mode == self._CONCATENATE:
            return concatenate([cell_output, attention_h])
        raise RuntimeError(  # already validated in __init__
            "got unexpected output_mode: {}".format(self.output_mode))

    @staticmethod
    def _num_elements(x):
        if hasattr(x, '__len__'):
            return len(x)
        else:
            return 1

    @property
    def _num_wrapped_states(self):
        return self._num_elements(self.cell.state_size)

    @property
    def _num_attention_states(self):
        return self._num_elements(self.attention_state_size)

    @property
    def _wrapped_cell_output_size(self):
        if hasattr(self.cell, "output_size"):
            return self.cell.output_size
        if hasattr(self.cell.state_size, '__len__'):
            return self.cell.state_size[0]
        return self.cell.state_size

    def build(self, input_shape):
        """Builds attention mechanism and wrapped cell (if keras layer).

        Arguments:
            input_shape: list of tuples of integers, the input feature shape
                (inputs sequence shape without time dimension) followed by
                constants (i.e. attended) shapes.
        """
        if not isinstance(input_shape, list):
            raise ValueError('input shape should contain shape of both cell '
                             'inputs and constants (attended)')

        attended_shape = input_shape[1:]
        input_shape = input_shape[0]
        self.attended_spec = [InputSpec(shape=shape) for shape in attended_shape]
        if isinstance(self.cell.state_size, int):
            cell_state_size = [self.cell.state_size]
        else:
            cell_state_size = list(self.cell.state_size)
        self.attention_build(
            input_shape=input_shape,
            cell_state_size=cell_state_size,
            attended_shape=attended_shape,
        )

        if isinstance(self.cell, Layer):
            if self.input_mode == self._REPLACE:
                cell_input_size = self._attention_size
            elif self.input_mode == self._CONCATENATE:
                cell_input_size = self.attention_size + input_shape[-1]
            else:
                raise RuntimeError(  # already validated in __init__
                    "got unexpected input_mode: {}".format(self.input_mode))

            cell_input_shape = (input_shape[0], cell_input_size)
            self.cell.build(cell_input_shape)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_size

    @property
    def trainable_weights(self):
        return (super(AttentionCellWrapper, self).trainable_weights +
                self.cell.trainable_weights)

    @property
    def non_trainable_weights(self):
        return (super(AttentionCellWrapper, self).non_trainable_weights +
                self.cell.non_trainable_weights)

    def get_config(self):
        config = {'attend_after': self.attend_after,
                  'input_mode': self.input_mode,
                  'output_mode': self.output_mode}

        cell_config = self.cell.get_config()
        config['cell'] = {'class_name': self.cell.__class__.__name__,
                          'config': cell_config}
        base_config = super(AttentionCellWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DenseAnnotationAttention(AttentionCellWrapper):
    # TODO update based on computing u externally!
    """Recurrent attention mechanism for attending sequences.

    This class implements the attention mechanism used in [1] for machine
    translation. It is, however, a generic sequence attention mechanism that can be
    used for other sequence-to-sequence problems.

    As any recurrent attention mechanism extending `_RNNAttentionCell`, this class
    should be used in conjunction with a wrapped (non attentive) RNN Cell, such as
    the `SimpleRNNCell`, `LSTMCell` or `GRUCell`. It modifies the input of the
    wrapped cell by attending to a constant sequence (i.e. independent of the time
    step of the recurrent application of the attention mechanism). The attention
    encoding is computed  by using a single hidden layer MLP which computes a
    weighting over the attended input. The MLP is applied to each time step of the
    attended, together with the previous state. The attention encoding is the taken
    as the weighted sum over the attended input using these weights.

    # Arguments
        cell: A RNN cell instance. The wrapped RNN cell wrapped by this attention
            mechanism. See docs of `cell` argument in the `RNN` Layer for further
            details.
        units: the number of hidden units in the single hidden MLP used for
            computing the attention weights.
        kernel_initializer: Initializer for all weights matrices
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for all bias vectors
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            all weights matrices. (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to all biases
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            all weights matrices. (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to all bias vectors
            (see [constraints](../constraints.md)).

     # Examples

    ```python
        # machine translation (similar to the architecture used in [1])
        x = Input((None,), name="input_sequences")
        y = Input((None,), name="target_sequences")
        x_emb = Embedding(INPUT_NUM_WORDS, 256, mask_zero=True)(x)
        y_emb = Embedding(TARGET_NUM_WORDS, 256, mask_zero=True)(y)
        encoder = Bidirectional(GRU(512, return_sequences=True))
        x_enc = encoder(x_emb)
        decoder = RNN(cell=DenseAnnotationAttention(cell=GRUCell(512), units=128),
                      return_sequences=True)
        h = decoder(y_emb, constants=x_enc)
        y_pred = TimeDistributed(Dense(TARGET_NUM_WORDS, activation='softmax'))(h)
        model = Model([y, x], y_pred)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=OPTIMIZER)
    ```

    # Details of attention mechanism
    Let {attended_1, ..., attended_I} denote the attended input sequence, where
    attended_i is the i:t attended input vector, h_cell_tm1 the previous state of
    the wrapped cell at the recurrent time step t. Then the attention encoding at
    time step t is computed as follows:

        e_i = MLP([attended_i, h_cell_tm1])  # [., .] denoting concatenation
        a_i = softmax_i({e_1, ..., e_I})
        attention_h_t = sum_i(a_i * h_i)

    # References
    [1] Neural Machine Translation by Jointly Learning to Align and Translate
        https://arxiv.org/abs/1409.0473
    """
    def __init__(self, cell,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DenseAnnotationAttention, self).__init__(cell, **kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def attention_call(self,
                       inputs,
                       cell_states,
                       attended,
                       attention_states,
                       attended_mask,
                       training=None):
        # there must be two attended sequences (verified in build)
        [attended, u] = attended
        attended_mask = attended_mask[0]
        h_cell_tm1 = cell_states[0]

        # compute attention weights
        w = K.repeat(K.dot(h_cell_tm1, self.W_a) + self.b_UW, K.shape(attended)[1])
        e = K.exp(K.dot(K.tanh(w + u), self.v_a) + self.b_v)

        if attended_mask is not None:
            e = e * K.cast(K.expand_dims(attended_mask, -1), K.dtype(e))

        # weighted average of attended
        a = e / K.sum(e, axis=1, keepdims=True)
        c = K.sum(a * attended, axis=1, keepdims=False)

        return c, [c]

    def attention_build(self, input_shape, cell_state_size, attended_shape):
        if not len(attended_shape) == 2:
            raise ValueError('There must be two attended tensors')
        for a in attended_shape:
            if not len(a) == 3:
                raise ValueError('only support attending tensors with dim=3')
        [attended_shape, u_shape] = attended_shape

        # NOTE _attention_size must always be set in `attention_build`
        self._attention_size = attended_shape[-1]
        units = u_shape[-1]

        kernel_kwargs = dict(initializer=self.kernel_initializer,
                             regularizer=self.kernel_regularizer,
                             constraint=self.kernel_constraint)
        self.W_a = self.add_weight(shape=(cell_state_size[0], units),
                                   name='W_a', **kernel_kwargs)
        self.v_a = self.add_weight(shape=(units, 1),
                                   name='v_a', **kernel_kwargs)

        bias_kwargs = dict(initializer=self.bias_initializer,
                           regularizer=self.bias_regularizer,
                           constraint=self.bias_constraint)
        self.b_UW = self.add_weight(shape=(units,),
                                    name="b_UW", **bias_kwargs)
        self.b_v = self.add_weight(shape=(1,),
                                   name="b_v", **bias_kwargs)

    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(DenseAnnotationAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
