from __future__ import print_function
import numpy as np
import keras
import theano.tensor as T
from input_data import read_data_sets
from lifelines.utils import concordance_index
import utils

class LDeepSurv(object):
	def __init__(self, input_shape, learning_rate = 0.001, 
        L2_reg = 0.0, L1_reg = 0.0, optimizer = 'adam', 
        hidden_layer_sizes = None, activation = 'relu', 
        batch_norm = False, dropout = 0.0, standardize = False):
        """
        This class implements and trains a DeepSurv model.
        Parameters:
            input_shape: shape of input. like (None, n)
            learning_rate: learning rate for training.
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
        X_input = keras.layers.Input(shape = input_shape, dtype = "float32", name = 'Input')
        X = X_input

        for n_layer in hidden_layer_sizes:
            # weight initialization methods for different activations
            if activation == 'relu':
                W_init = 'glorot_uniform'
            else:
                # TODO:
                W_init = 'glorot_uniform'

            # Z = w * x + b
            # weight initialize and regularize
            Z = keras.layers.Dense(units = n_layer, 
                                   kernel_initializer = W_init,
                                   kernel_regularize = keras.regularizers.l1_l2(L1_reg, L2_reg))(X)

            # batch Norm should be added before Activation
            # the advantages of BatchNorm
            if batch_norm:
                Z = keras.layers.BatchNormalization()(Z)

            # activation function
            A = keras.layers.Activation(activation)(Z)

            # Dropout
            if dropout != .0:
                A = keras.layers.Dropout(dropout)(A)

            X = A

        # output
        output = keras.layer.Dense(units = 1,
                                   kernel_initializer = 'glorot_uniform',
                                   kernel_regularize = keras.regularizers.l1_l2(L1_reg, L2_reg))(X)

        # model
        self.model = keras.models.Model(inputs = X_input, outputs = output, name = 'LDeepSurv')
        
        # optimizer
        if optimizer == 'adam':
            opt = keras.optimizers.Adam()
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(lr = learning_rate, momentum = 0.9, decay = 0.0, nesterov = False)
        else:
            opt = keras.optimizers.Adam()

        self.model.compile(optimizer = opt, loss = self._negative_log_likelihood, metrics = [self._CI])

        # Store and set needed Hyper-parameters for tuning
        self.hyperparams = {
            'learning_rate': learning_rate,
            'hidden_layers_sizes': hidden_layers_sizes,
            'optimizer': optimizer,
            'L2_reg': L2_reg,
            'L1_reg': L1_reg,
            'activation': activation,
            'dropout': dropout,
            'batch_norm': batch_norm,
            'standardize': standardize
        }

        # Store needed parameters for training or others
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.standardize = standardize
        self.restored_update_params = None

    def _negative_log_likelihood(self, y_true, y_pred):
        """
        Callable loss function for DeepSurv network.

        Parameters:
            y_true: labels(dict like {'E': ,'T': }) for input_X.
            y_pred: outputs of the network for input_X.

        Returns:
            loss value formatted as theano expression.
        """
        # get information from labels for next calculation.
        (ties, E, T, failures, atrisk) = utils.decode_ET(y_true)
        logL = 0
        if ties == 'noties':
            hazard_ratio = T.exp(y_pred)
            log_risk = T.log(T.extra_ops.cumsum(hazard_ratio))
            likelihood = y_pred.T - log_risk
            uncensored_likelihood = likelihood * E
            logL = -T.sum(uncensored_likelihood)
        else:
            for t in failures:
                tfail = failures[t]
                trisk = atrisk[t]
                d = len(tfail)

                logL += -T.sum(y_pred[tfail])

                if ties == 'breslow':
                    s = T.sum(T.exp(y_pred[trisk]))
                    logL += T.log(s)*d
                elif ties == 'efron':
                    s = T.sum(T.exp(y_pred[trisk]))
                    r = T.sum(T.exp(y_pred[tfail]))
                    for j in range(d):
                        logL += T.log(s - j * r / d)
                else:
                    raise NotImplementedError('tie breaking method not recognized')

        return logL
    
    def _CI(self, y_true, y_pred):
        """
        Compute the concordance-index value.
        Parameters:
            y_true: labels(dict like {'e': ,'t': }) for input_X.
            y_pred: outputs of the network for input_X.

        Returns:
            concordance index.
        """
        hr_pred = -np.exp(y_pred)
        ci = concordance_index(y_true['t'], hr_pred, y_true['e'])
        return ci

    def train(self, data_dir, vr = 0.2, batch_size = 1000, epoch = 10000):

        # DataSets = read_data_sets(train_dir = data_dir, validation_ratio = vr)
        # train_data, validation_data, test_data = DataSets.train, DataSets.validation, DataSets.test

        # for num_iters in range(epoch):

        #     for i in range(train_data.num_examples / batch_size):
        #         batch_data = train_data.next_batch(batch_size = batch_size)