import keras

class LDeepSurv(object):
	def __init__(self, input_shape, learning_rate = 0.001, lr_decay = 0.0, 
        L2_reg = 0.0, L1_reg = 0.0, optimizer = 'Adam', 
        hidden_layer_sizes = None, activation = 'relu', 
        batch_norm = False, dropout = 0.0, standardize = False):
        """
        This class implements and trains a DeepSurv model.
        Parameters:
            input_shape: shape of input. like (None, n)
            learning_rate: learning rate for training.
            lr_decay: coefficient for Power learning rate decay.
            L2_reg: coefficient for L2 weight decay regularization. Used to help
                prevent the model from overfitting.
            L1_reg: coefficient for L1 weight decay regularization
            optimizer: optimizer for backpropogation
            hidden_layer_sizes: a list of integers to determine the size of
                each hidden layer.
            activation: activation name from keras.
                Default: 'relu'
            batch_norm: True or False. Include batch normalization layers.
            dropout: if 0.0, the percentage of dropout to include
                after each hidden layer. Default: 0.0
            standardize: True or False. Include standardization layer after
                input layer.
        """
        X = keras.layers.Input(shape = input_shape, dtype = "float32", name = 'Input')

        for n_layer in hidden_layer_sizes:
            # weight initialization methods for different activations
            if activation == 'relu':
                W_init = 'glorot_uniform'
            else:
                # TODO:
                W_init = 'glorot_uniform'

            # X = w * x + b
            # weight initialize and regularize
            X = keras.layers.Dense(units = n_layer, 
                                   kernel_initializer = W_init,
                                   kernel_regularize = keras.regularizers.l1_l2(L1_reg, L2_reg))(X)

            # batch Norm should be added before Activation
            # the advantages of BatchNorm
            if batch_norm:
                X = keras.layers.BatchNormalization()(X)

            # activation function
            X = keras.layers.Activation(activation)

            # Dropout
            if dropout != .0:
                X = keras.layers.Dropout(dropout)(X)

        # output
        output = keras.layer.Dense(units = 1,
                              kernel_initializer = 'glorot_uniform',
                              kernel_regularize = keras.regularizers.l1_l2(L1_reg, L2_reg))(X)

        self.output = output
        # Store and set needed Hyper-parameters for tuning
        self.hyperparams = {
            'learning_rate': learning_rate,
            'hidden_layers_sizes': hidden_layers_sizes,
            'lr_decay': lr_decay,
            'optimizer': optimizer,
            'L2_reg': L2_reg,
            'L1_reg': L1_reg,
            'activation': activation,
            'dropout': dropout,
            'batch_norm': batch_norm,
            'standardize': standardize
        }

        # Store needed parameters for trainning or others
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.optimizer = optimizer
        self.standardize = standardize
        self.restored_update_params = None

    def loss_function(self):
        