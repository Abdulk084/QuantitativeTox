from tensorflow.keras import activations, backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from spektral.layers.convolutional.message_passing import MessagePassing


class GINConv(MessagePassing):
    r"""
    A Graph Isomorphism Network (GIN) as presented by
    [Xu et al. (2018)](https://arxiv.org/abs/1810.00826).

    **Mode**: single, disjoint.

    **This layer expects a sparse adjacency matrix.**

    This layer computes for each node \(i\):
    $$
        \Z_i = \textrm{MLP}\big( (1 + \epsilon) \cdot \X_i + \sum\limits_{j \in \mathcal{N}(i)} \X_j \big)
    $$
    where \(\textrm{MLP}\) is a multi-layer perceptron.

    **Input**

    - Node features of shape `(N, F)`;
    - Binary adjacency matrix of shape `(N, N)`.

    **Output**

    - Node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `epsilon`: unnamed parameter, see
    [Xu et al. (2018)](https://arxiv.org/abs/1810.00826), and the equation above.
    By setting `epsilon=None`, the parameter will be learned (default behaviour).
    If given as a value, the parameter will stay fixed.
    - `mlp_hidden`: list of integers, number of hidden units for each hidden
    layer in the MLP (if None, the MLP has only the output layer);
    - `mlp_activation`: activation for the MLP layers;
    - `activation`: activation function to use;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the weights;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the weights;
    - `bias_constraint`: constraint applied to the bias vector.
    """

    def __init__(self,
                 channels,
                 epsilon=None,
                 mlp_hidden=None,
                 mlp_activation='relu',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(aggregate='sum',
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         **kwargs)
        self.channels = self.output_dim = channels
        self.epsilon = epsilon
        self.mlp_hidden = mlp_hidden if mlp_hidden else []
        self.mlp_activation = activations.get(mlp_activation)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint
        )

        self.mlp = Sequential([
            Dense(channels, self.mlp_activation, **layer_kwargs)
            for channels in self.mlp_hidden
        ] + [Dense(self.channels, self.activation, use_bias=self.use_bias, **layer_kwargs)])

        if self.epsilon is None:
            self.eps = self.add_weight(shape=(1,),
                                       initializer='zeros',
                                       name='eps')
        else:
            # If epsilon is given, keep it constant
            self.eps = K.constant(self.epsilon)

        self.built = True

    def call(self, inputs):
        X, A, E = self.get_inputs(inputs)
        output = self.mlp((1.0 + self.eps) * X + self.propagate(X, A, E))

        return output

    def get_config(self):
        config = {
            'channels': self.channels,
            'epsilon': self.epsilon,
            'mlp_hidden': self.mlp_hidden,
            'mlp_activation': self.mlp_activation
        }
        base_config = super().get_config()
        base_config.pop('aggregate')  # Remove it because it's defined by constructor
        return {**base_config, **config}
