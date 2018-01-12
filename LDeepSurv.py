from __future__ import print_function
import numpy as np
import tensorflow as tf
from lifelines.utils import concordance_index
import utils


class LDeepSurv(object):
    def __init__(self, input_node, hidden_layers_node, output_node,
        learning_rate = 0.001, learning_rate_decay = 1.0, 
        activation = 'relu', 
        L2_reg = 0.0, L1_reg = 0.0, optimizer = 'sgd', 
        dropout = 0.0):
        # Data input 
        self.X = tf.placeholder(tf.float32, [None, input_node], name = 'x-Input')
        self.y_ = tf.placeholder(tf.float32, [None, output_node], name = 'label-Input')
        # hidden layers
        prev_node = input_node
        prev_x = self.X
        for i in range(len(hidden_layers_node)):
            with tf.variable_scope('layer' + str(i+1)):
                weights = self.get_weight_variable([prev_node, hidden_layers_node[i]], L1_reg, L2_reg)
                biases = tf.get_variable('biases', [hidden_layers_node[i]],
                                         initializer=tf.constant_initializer(0.0))
                if activation == 'relu':
                    layer_out = tf.nn.relu(tf.matmul(prev_x, weights) + biases)
                elif activation == 'sigmoid':
                    layer_out = tf.nn.sigmoid(tf.matmul(prev_x, weights) + biases)
                elif activation == 'tanh':
                    layer_out = tf.nn.tanh(tf.matmul(prev_x, weights) + biases)
                else:
                    layer_out = tf.nn.relu(tf.matmul(prev_x, weights) + biases)
                prev_node = hidden_layers_node[i]
                prev_x = layer_out
        # output layers
        with tf.variable_scope('layer_last'):
            weights = self.get_weight_variable([prev_node, output_node], L1_reg, L2_reg)
            biases = tf.get_variable('biases', [output_node],
                                     initializer=tf.constant_initializer(0.0))
            layer_out = tf.matmul(prev_x, weights) + biases
        self.y = layer_out
        self.configuration = {
            'input_node': input_node,
            'hidden_layers_node': hidden_layers_node,
            'output_node': output_node,
            'learning_rate': learning_rate,
            'learning_rate_decay': learning_rate_decay,
            'activation': activation,
            'L1_reg': L1_reg,
            'L2_reg': L2_reg,
            'optimizer': optimizer,
            'dropout': dropout
        }

    def train(self, X, label, num_epoch=1000):
        """
        train DeepSurv network
        Parameters:
            X: np.array[N, m]
            label: dict
                   e: np.array[N]
                   t: np.array[N]
        """
        global_step = tf.Variable(0, trainable=False)
        # Batch contain all train dataset
        learning_rate = tf.train.exponential_decay(
                self.configuration['learning_rate'],
                global_step,
                1,
                self.configuration['learning_rate_decay']
            )
        # loss value
        loss_fun = self._negative_log_likelihood(self.y_, self.y)
        loss = loss_fun + tf.add_n(tf.get_collection('losses'))
        # SGD Optimizer
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        # train
        n = label['e'].shape[0]
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            for i in range(num_epoch):
                _, output_y, loss_value, step = sess.run([train_step, self.y, loss, global_step],
                                               feed_dict = {self.X: X, self.y_: label['e'].reshape((n, 1))})
                if i % 100 == 0:
                    print("-------------------------------------------------")
                    print("training steps %d: loss=%g.\n" % (step, loss_value))
                    print("CI on train set: %g.\n" % self._Metrics_CI(label, output_y))

    def get_weight_variable(self, shape, L1_reg, L2_reg):
        weights = tf.get_variable('weights', shape, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        if L1_reg != 0.0:
            tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(L1_reg)(weights))

        if L2_reg != 0.0:
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(L2_reg)(weights))
        return weights

    def _negative_log_likelihood(self, y_true, y_pred):
        """
        Callable loss function for DeepSurv network.
        """
        hazard_ratio = tf.exp(y_pred)
        log_risk = tf.log(tf.cumsum(hazard_ratio))
        likelihood = y_pred - log_risk
        # dimension for E: np.array -> [None, 1]
        uncensored_likelihood = likelihood * y_true
        logL = -tf.reduce_sum(uncensored_likelihood)
        return logL
    
    def _Metrics_CI(self, label_true, y_pred):
        """
        Compute the concordance-index value.
        """
        hr_pred = -np.exp(y_pred)
        ci = concordance_index(label_true['t'],
                               hr_pred,
                               label_true['e'])
        return ci