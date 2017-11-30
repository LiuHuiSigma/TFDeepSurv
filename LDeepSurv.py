import keras

class LDeepSurv(object):
	def __init__(self, n_input, learning_rate, lr_decay, L2_reg, L1_reg, optimizer, 
		hidden_layer_sizes, activation, batch_norm, dropout, standardize):
	"""
    This class implements and trains a DeepSurv model.
    Parameters:
        n_input: number of input nodes.
        learning_rate: learning rate for training.
        lr_decay: coefficient for Power learning rate decay.
        L2_reg: coefficient for L2 weight decay regularization. Used to help
            prevent the model from overfitting.
        L1_reg: coefficient for L1 weight decay regularization
        optimizer: optimizer for backpropogation
        hidden_layer_sizes: a list of integers to determine the size of
            each hidden layer.
        activation: a lasagne activation class.
            Default: lasagne.nonlinearities.rectify
        batch_norm: True or False. Include batch normalization layers.
        dropout: if not None or 0, the percentage of dropout to include
            after each hidden layer. Default: None
        standardize: True or False. Include standardization layer after
            input layer.
    """