from __future__ import print_function, division

from keras import backend as K
from keras.engine import InputSpec
from keras.engine import Layer
from keras.layers.merge import concatenate
from keras.layers.recurrent import LSTM as _LSTM

from extkeras.functional import FunctionalBlock
from extkeras import initializers


class LSTM(_LSTM):
    """Modification of base class that:
        1) breaks apart 'step' function (for easier modification of output in
           extensions of this class)
        2) adds the option of concatenate cell activation (c) to h in output
           (if one wants to use cell sequence for later layers and also nice
           for debuging and testing)

    # Args
    output_cells (bool): if True output wil be concatenation of h and c
        default value (False) gives identical behaviour to base class.
    """
    def __init__(
        self,
        output_cells=False,
        **kwargs
    ):
        super(LSTM, self).__init__(**kwargs)
        self.output_cells = output_cells

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if self.output_cells:
            out_units = self.units * 2
        else:
            out_units = self.units

        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1], out_units)
        else:
            output_shape = (input_shape[0], out_units)

        if self.return_state:
            state_shape = [(input_shape[0], self.units) for _ in self.states]
            return [output_shape] + state_shape
        else:
            return output_shape

    def step(self, inputs, states):
        h, c = self._get_hc(inputs, states)
        if self.output_cells:
            output = concatenate([h, c])
        else:
            output = h

        if 0 < self.dropout + self.recurrent_dropout:
            output._uses_learning_phase = True

        return output, [h, c]

    def _get_ifco(self, inputs, states):
        """Break out of computation of gates and cell for more flexible reuse.
        """
        h_tm1 = states[0]
        c_tm1 = states[1]
        dp_mask = states[2]
        rec_dp_mask = states[3]

        if self.implementation == 2:
            z = K.dot(inputs * dp_mask[0], self.kernel)
            z += K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]
            z2 = z[:, 2 * self.units: 3 * self.units]
            z3 = z[:, 3 * self.units:]

            i = self.recurrent_activation(z0)
            f = self.recurrent_activation(z1)
            c = f * c_tm1 + i * self.activation(z2)
            o = self.recurrent_activation(z3)
        else:
            if self.implementation == 0:
                x_i = inputs[:, :self.units]
                x_f = inputs[:, self.units: 2 * self.units]
                x_c = inputs[:, 2 * self.units: 3 * self.units]
                x_o = inputs[:, 3 * self.units:]
            elif self.implementation == 1:
                x_i = K.dot(inputs * dp_mask[0], self.kernel_i) + self.bias_i
                x_f = K.dot(inputs * dp_mask[1], self.kernel_f) + self.bias_f
                x_c = K.dot(inputs * dp_mask[2], self.kernel_c) + self.bias_c
                x_o = K.dot(inputs * dp_mask[3], self.kernel_o) + self.bias_o
            else:
                raise ValueError('Unknown `implementation` mode.')

            i = self.recurrent_activation(x_i + K.dot(h_tm1 * rec_dp_mask[0],
                                                      self.recurrent_kernel_i))
            f = self.recurrent_activation(x_f + K.dot(h_tm1 * rec_dp_mask[1],
                                                      self.recurrent_kernel_f))
            c = f * c_tm1 + i * self.activation(
                x_c + K.dot(h_tm1 * rec_dp_mask[2],
                            self.recurrent_kernel_c))
            o = self.recurrent_activation(x_o + K.dot(h_tm1 * rec_dp_mask[3],
                                                      self.recurrent_kernel_o))

        return i, f, c, o

    def _get_hc(self, inputs, states):
        i, f, c, o = self._get_ifco(inputs, states)
        h = o * self.activation(c)

        return h, c


class CellMaskedLSTM(LSTM):
    """ Modification of base LSTM implementation to allow for flexible masking
    of cells at specific timesteps. This way we can externally/separately
    define or learn a model that blocks part of cells to be updated and
    propagate memory further. A special case of this is the Phased-LSTM,
    see: TODO
    """

    def __init__(self, *args, **kwargs):
        super(CellMaskedLSTM, self).__init__(*args, **kwargs)
        if self.recurrent_dropout != 0 or self.dropout != 0:
            raise NotImplementedError('')
            # TODO fix get_constants to handle dropout

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        lstm_input_shape = input_shape[:2] + (input_shape[2] - self.units, )
        super(CellMaskedLSTM, self).build(lstm_input_shape)

        self.input_spec = [InputSpec(shape=input_shape)]

    def preprocess_input(self, inputs, training=None):
        if self.implementation == 0:
            cell_mask = inputs[:, :, -self.units:]
            inputs = inputs[:, :, :-self.units]

            inputs_prep = super(CellMaskedLSTM, self).preprocess_input(
                inputs,
                training
            )
            return K.concatenate([inputs_prep, cell_mask], axis=2)
        else:
            return inputs

    def _get_hc(self, inputs, states):
        cell_mask = inputs[:, -self.units:]
        inputs = inputs[:, :-self.units]
        i, f, c_unmasked, o = self._get_ifco(inputs, states)
        c_tm1 = states[1]
        c = c_unmasked * cell_mask + (1 - cell_mask) * c_tm1
        h = o * self.activation(c)

        return h, c


class PhasedLSTMCellMask(Layer):
    """Simple feed-fwd layer that produces a "LSTM cell-mask" (to be used
    with CellMaskedLSTM) and thereby implementing the "Phased" part PhasedLSTM.

    NOTE the call part of this code copied from:
    https://github.com/fferroni/PhasedLSTM-Keras

    TODO complete docs!

    # References
    - [Phased LSTM: Accelerating Recurrent Network Training for Long or
        Event-based Sequences](https://arxiv.org/abs/1610.09513)
    """

    def __init__(
        self,
        units,
        initializer=None,
        alpha=0.001,
        **kwargs
    ):
        self.units = units
        self.input_dim = 1
        self.timegate = None  # set in build
        self.alpha = alpha
        self.initializer = initializer or initializers.TimeGate()
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(PhasedLSTMCellMask, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        assert input_dim == 1
        self.input_dim = input_dim
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.timegate = self.add_weight(
            shape=(3, self.units),
            initializer=self.initializer,
            name='timegate',
            # regularizer=self.regularizer,
            # constraint=self.constraint
        )
        self.built = True

    def call(self, inputs, mask=None):
        t = inputs
        timegate = K.abs(self.timegate)
        period = timegate[0]
        shift = timegate[1]
        r_on = timegate[2]

        phi = ((t - shift) % period) / period
        # K.switch not consistent between Theano and Tensorflow backend,
        # so write explicitly.
        # TODO check if still the case
        up = K.cast(K.less(phi, r_on * 0.5), K.floatx()) * 2 * phi / r_on
        mid = (
            K.cast(K.less(phi, r_on), K.floatx()) *
            K.cast(K.greater(phi, r_on * 0.5), K.floatx()) *
            (2 - (2 * phi / r_on))
        )
        end = K.cast(K.greater(phi, r_on * 0.5), K.floatx()) * self.alpha * phi
        k = up + mid + end

        return k

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {'output_dim': self.units, 'alpha': self.alpha}
        base_config = super(PhasedLSTMCellMask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PhasedLSTM(FunctionalBlock):

    def __init__(
        self,
        units,
        regular_units=0,
        alpha=0.001,
        cell_mask_initializer=None,
        **kwargs
    ):
        if regular_units != 0:
            raise NotImplementedError('')
            # TODO we can "spare" some units and let them operate as usual
            # by just masking this subset with ones

        name = kwargs.pop('name', self.__class__.__name__)
        self.cell_mask = PhasedLSTMCellMask(
            name=name+'_PhasedLSTMCellMask',
            units=units,
            initializer=cell_mask_initializer,
            alpha=alpha
        )
        self.cell_masked_lstm = CellMaskedLSTM(
            name=name+'_CellMaskedLSTM',
            units=units,
            **kwargs
        )

    def __call__(self, inputs):
        lstm_inputs, time = inputs
        cell_mask = self.cell_mask(time)
        outputs = self.cell_masked_lstm(
            concatenate([lstm_inputs, cell_mask], axis=2)
        )

        return outputs
