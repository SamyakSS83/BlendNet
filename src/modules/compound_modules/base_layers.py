import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_ACTIVATION_MAP = {'ReLU', 'Sigmoid', 'Tanh', 'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus', 'SiLU', 'None'}
EPS = 1e-5

def get_activation(activation):
    """ returns the activation function represented by the input string """
    if activation and callable(activation):
        # activation is already a function
        return activation
    # search in SUPPORTED_ACTIVATION_MAP a torch.nn.modules.activation
    activation = [x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) == 1 and isinstance(activation[0], str), 'Unhandled activation function'
    activation = activation[0]
    if activation.lower() == 'none':
        return None
    return vars(torch.nn.modules.activation)[activation]()


class FCLayer(nn.Module):
    r"""
    A simple fully connected and customizable layer. This layer is centered around a torch.nn.Linear module.
    The order in which transformations are applied is:
    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)
    Arguments
    ----------
        in_dim: int
            Input dimension of the layer (the torch.nn.Linear)
        out_dim: int
            Output dimension of the layer.
        dropout: float, optional
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable, optional
            Activation function to use.
            (Default value = relu)
        batch_norm: bool, optional
            Whether to use batch normalization
            (Default value = False)
        bias: bool, optional
            Whether to enable bias in for the linear layer.
            (Default value = True)
        init_fn: callable, optional
            Initialization function to use for the weight of the layer. Default is
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{in_dim}}`
            (Default value = None)
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        activation='relu',
        dropout=0.,
        batch_norm=False,
        bias=True,
        init_fn=None,
        device=None,
        dtype=None
    ):
        super(FCLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        if activation is not None:
            self.activation = get_activation(activation)
        else:
            self.activation = None
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.batch_norm = nn.BatchNorm1d(out_dim) if batch_norm else None
        self.init_fn = nn.init.xavier_uniform_ if init_fn is None else init_fn
        self.init_parameters()

    def init_parameters(self):
        """ Initialize the parameters if not done by PyTorch already """
        if self.init_fn is not None:
            self.init_fn(self.linear.weight)
            if self.bias:
                nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        """
        Forward pass through the layer.
        Parameters
        ----------
        inputs : torch.Tensor
            the input tensor

        Returns
        -------
        tensor : torch.Tensor
            the transformed tensor
        """
        hidden = self.linear(inputs)
        if self.activation is not None:
            hidden = self.activation(hidden)
        if self.dropout is not None:
            hidden = self.dropout(hidden)
        if self.batch_norm is not None:
            if len(hidden.shape) == 3:
                hidden_bn = hidden.reshape(-1, hidden.shape[2])
                hidden_bn = self.batch_norm(hidden_bn)
                hidden = hidden_bn.reshape(hidden.shape)
            else:
                hidden = self.batch_norm(hidden)
        return hidden

    def __repr__(self):
        return "FCLayer(in_dim=%s, out_dim=%s, bias=%s)" % (
            self.in_dim, self.out_dim, self.bias)


class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    Parameters
    ----------
    layer_sizes: list
        List of layer sizes.
    dropout: float
        Probability of dropping a weight.
    activation: str
        Name of activation function. Valid activations are found in SUPPORTED_ACTIVATION_MAP.
    batch_norm: bool
        Whether batch normalization is applied.
    """

    def __init__(
        self,
        layer_sizes,
        dropout=0,
        activation='ReLU',
        batch_norm=False,
        last_layer_activation='None',
    ):
        super(MLP, self).__init__()

        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            if i == self.n_layers - 1:  # last layer has potentially different activation
                self.layers.append(
                    FCLayer(
                        layer_sizes[i],
                        layer_sizes[i + 1],
                        activation=last_layer_activation,
                        dropout=dropout,
                        batch_norm=batch_norm,
                    )
                )
            else:
                self.layers.append(
                    FCLayer(
                        layer_sizes[i],
                        layer_sizes[i + 1],
                        activation=activation,
                        dropout=dropout,
                        batch_norm=batch_norm,
                    )
                )

    def forward(self, inputs):
        """
        Forward pass through the MLP.
        Parameters
        ----------
        inputs : torch.Tensor
            the input tensor

        Returns
        -------
        tensor : torch.Tensor
            the transformed tensor
        """
        if len(inputs.shape) > 2:
            inputs = inputs.view(-1, inputs.shape[-1])
        h = inputs
        for layer in self.layers:
            h = layer(h)
        return h

    def __repr__(self):
        return "MLP(layer_sizes=%s, dropout=%s, activation=%s, batch_norm=%s)" % (
            self.layer_sizes, self.dropout, self.activation, self.batch_norm)
