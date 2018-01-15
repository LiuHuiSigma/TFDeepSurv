from __future__ import print_function
import numpy as np
import tensorflow as tf
from lifelines.utils import concordance_index
import vision

class LDeepSurv(object):
    def __init__(self, input_node, hidden_layers_node, output_node,
        learning_rate = 0.001, learning_rate_decay = 1.0, 
        activation = 'tanh', 
        L2_reg = 0.0, L1_reg = 0.0, optimizer = 'sgd', 
        dropout_keep_prob = 1.0):
        # Data input 
        self.X = tf.placeholder(tf.float32, [None, input_node], name = 'x-Input')
        self.y_ = tf.placeholder(tf.float32, [None, output_node], name = 'label-Input')
        # hidden layers
        self.nnweights = []
        prev_node = input_node
        prev_x = self.X
        for i in range(len(hidden_layers_node)):
            layer_name = 'layer' + str(i+1)
            with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                weights = tf.get_variable('weights', [prev_node, hidden_layers_node[i]], 
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
                self.nnweights.append(weights)
                biases = tf.get_variable('biases', [hidden_layers_node[i]],
                                         initializer=tf.constant_initializer(0.0))
                layer_out = tf.nn.dropout(tf.matmul(prev_x, weights) + biases, dropout_keep_prob)
                if activation == 'relu':
                    layer_out = tf.nn.relu(layer_out)
                elif activation == 'sigmoid':
                    layer_out = tf.nn.sigmoid(layer_out)
                elif activation == 'tanh':
                    layer_out = tf.nn.tanh(layer_out)
                else:
                    layer_out = tf.nn.tanh(layer_out)
                prev_node = hidden_layers_node[i]
                prev_x = layer_out
        # output layers
        layer_name = 'layer_last'
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable('weights', [prev_node, output_node], 
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.nnweights.append(weights)
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
            'dropout': dropout_keep_prob
        }
        # create new Session
        self.sess = tf.Session()

    def train(self, X, label, 
              num_epoch=5000, iteration=-1, 
              plot_train_loss=False, plot_train_CI=False):
        """
        train DeepSurv network
        Parameters:
            X: np.array[N, m]
            label: dict
                   e: np.array[N]
                   t: np.array[N]
            num_epoch: times of iterating whole train set.
            iteration: print information on train set every iteration train steps.
                       default -1, keep silence.
            plot_train_loss: plot curve of loss value during training.
            plot_train_CI: plot curve of CI on train set during training.
        """
        # global step
        with tf.variable_scope('training_step', reuse=tf.AUTO_REUSE):
            global_step = tf.get_variable("global_step", [], 
                                          dtype=tf.int32,
                                          initializer=tf.constant_initializer(0), 
                                          trainable=False)
        # loss value
        reg_item = tf.contrib.layers.l1_l2_regularizer(self.configuration['L1_reg'],
                                                       self.configuration['L2_reg'])
        reg_term = tf.contrib.layers.apply_regularization(reg_item, self.nnweights)
        loss_fun = self._negative_log_likelihood(self.y_, self.y)
        loss = loss_fun + reg_term
        # SGD Optimizer
        if self.configuration['optimizer'] == 'sgd':
            learning_rate = tf.train.exponential_decay(
                self.configuration['learning_rate'],
                global_step,
                1,
                self.configuration['learning_rate_decay']    
            )
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        elif self.configuration['optimizer'] == 'adam':
            train_step = tf.train.GradientDescentOptimizer(self.configuration['learning_rate']).\
                                                           minimize(loss, global_step=global_step)            
        else:
            train_step = tf.train.GradientDescentOptimizer(self.configuration['learning_rate']).\
                                                           minimize(loss, global_step=global_step)            
        # record training steps
        loss_list = []
        CI_list = []
        # train steps
        n = label['e'].shape[0]
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        for i in range(num_epoch):
            _, output_y, loss_value, step = self.sess.run([train_step, self.y, loss, global_step],
                                                          feed_dict = {self.X: X, self.y_: label['e'].reshape((n, 1))})
            # record information
            loss_list.append(loss_value)
            CI = self._Metrics_CI(label, output_y)
            CI_list.append(CI)
            if (iteration != -1) and (i % iteration == 0):
                print("-------------------------------------------------")
                print("training steps %d:\nloss = %g.\n" % (step, loss_value))
                print("CI = %g.\n" % CI)
        # plot curve
        if plot_train_loss:
            vision.plot_train_curve(loss_list, title="Loss(train)")
        if plot_train_CI:
            vision.plot_train_curve(CI_list, title="CI(train)")

    def predict(self, X):
        """
        Predict risk of X using trained network.
        """
        risk = self.sess.run([self.y], feed_dict = {self.X: X})
        return np.squeeze(risk)

    def eval(self, X, label):
        """
        Evaluate test set using CI metrics.
        """
        pred_risk = self.predict(X)
        CI = self._Metrics_CI(label, pred_risk)
        print("CI on test set = %g." % CI)
        return CI

    def close(self):
        self.sess.close()
        print("Current session closed!")
    
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

    def evaluate_var_byWeights(self):
        # fetch weights of network
        W = [self.sess.run(w) for w in self.nnweights]
        n_w = len(W)
        # matrix multiplication for all hidden layers except last output layer
        hiddenMM = W[- 2].T
        for i in range(n_w - 3, -1, -1):
            hiddenMM = np.dot(hiddenMM, W[i].T)
        # multiply last layer matrix and compute the sum of each varible for VIP
        last_layer = W[-1]
        s = np.dot(np.diag(last_layer[:, 0]), hiddenMM)

        sumr = s / s.sum(axis=1).reshape(s.shape[0] ,1)
        score = sumr.sum(axis=0)
        VIP = score / score.max()
        for i, v in enumerate(VIP):
            print("feature %d score : %g." % (i, v))
        return VIP