from __future__ import print_function
import numpy as np
import tensorflow as tf
from lifelines.utils import concordance_index
from supersmoother import SuperSmoother

import vision, utils

class L2DeepSurv(object):
    def __init__(self, X, label,
        input_node, hidden_layers_node, output_node,
        learning_rate=0.001, learning_rate_decay=1.0, 
        activation='tanh', 
        L2_reg=0.0, L1_reg=0.0, optimizer='sgd', 
        dropout_keep_prob=1.0):
        # prepare data
        self.train_data = {}
        self.train_data['X'], self.train_data['E'], \
            self.train_data['T'], self.train_data['failures'], \
            self.train_data['atrisk'], self.train_data['ties'] = utils.parse_data(X, label)
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

    def train(self, num_epoch=5000, iteration=-1, 
              seed = 1, 
              plot_train_loss=False, plot_train_CI=False):
        """
        train DeepSurv network
        Parameters:
            num_epoch: times of iterating whole train set.
            iteration: print information on train set every iteration train steps.
                       default -1, keep silence.
            seed: set random state.
            plot_train_loss: plot curve of loss value during training.
            plot_train_CI: plot curve of CI on train set during training.
        """
        # Set random state
        tf.set_random_seed(seed)
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
        N = self.train_data['E'].shape[0]
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        for i in range(num_epoch):
            _, output_y, loss_value, step = self.sess.run([train_step, self.y, loss, global_step],
                                                          feed_dict = {self.X:  self.train_data['X'],
                                                                       self.y_: self.train_data['E'].reshape((N, 1))})
            # record information
            loss_list.append(loss_value)
            label = {'t': self.train_data['T'],
                     'e': self.train_data['E']}
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
        return CI

    def close(self):
        self.sess.close()
        print("Current session closed!")
    
    def _negative_log_likelihood(self, y_true, y_pred):
        """
        Callable loss function for DeepSurv network.
        """
        logL = 0
        # pre-calculate cumsum
        cumsum_y_pred = tf.cumsum(y_pred)
        hazard_ratio = tf.exp(y_pred)
        cumsum_hazard_ratio = tf.cumsum(hazard_ratio)
        if self.train_data['ties'] == 'noties':
            log_risk = tf.log(cumsum_hazard_ratio)
            likelihood = y_pred - log_risk
            # dimension for E: np.array -> [None, 1]
            uncensored_likelihood = likelihood * y_true
            logL = -tf.reduce_sum(uncensored_likelihood)
        else:
            # Loop for death times
            for t in self.train_data['failures']:                                                                       
                tfail = self.train_data['failures'][t]
                trisk = self.train_data['atrisk'][t]
                d = len(tfail)
                dr = len(trisk)

                logL += -cumsum_y_pred[tfail[-1]] + (0 if tfail[0] == 0 else cumsum_y_pred[tfail[0]-1])

                if self.train_data['ties'] == 'breslow':
                    cumsum_hazard_ratio
                    s = cumsum_hazard_ratio[trisk[-1]]
                    logL += tf.log(s) * d
                elif self.train_data['ties'] == 'efron':
                    s = cumsum_hazard_ratio[trisk[-1]]
                    r = cumsum_hazard_ratio[tfail[-1]] - (0 if tfail[0] == 0 else cumsum_hazard_ratio[tfail[0]-1])
                    for j in range(d):
                        logL += tf.log(s - j * r / d)
                else:
                    raise NotImplementedError('tie breaking method not recognized')

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
        """
        evaluate feature importance by weights of NN 
        """
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
            print("%dth feature score : %g." % (i, v))
        return VIP

    def survivalRate(self, X, algo="wwe", base_X=None, base_label=None, smoothed=False):
        """
        Evaluate survival rate.
        """
        risk = self.predict(X)
        hazard_ratio = np.exp(risk.reshape((risk.shape[0], 1)))
        # Estimate S0(t) using data(base_X, base_label)
        T0, S0 = self.basesurv(algo=algo, X=base_X, label=base_label, smoothed=smoothed)
        ST = S0**(hazard_ratio)

        vision.plt_surLines(T0, ST)

        return T0, ST

    def basesurv(self, algo="wwe", X=None, label=None, smoothed=False):
        """
        Algorithm for estimating S0:
        (1). wwe: WWE(with ties)
        (2). kp: Kalbfleisch & Prentice Estimator(without ties)
        (2). bsl: breslow(with ties, but exists negative value)
        Estimate base survival function S0(t) based on data(X, label).
        """
        # Get data for estimating S0(t)
        if X is None or label is None:
            X = self.train_data['X']
            label = {'t': self.train_data['T'],
                     'e': self.train_data['E']}
        X, E, T, failures, atrisk, ties = utils.parse_data(X, label)

        s0 = [1]
        risk = self.predict(X)
        hz_ratio = np.exp(risk)
        if algo == 'wwe':        
            for t in T[::-1]:
                if t in atrisk:
                    # R(t_i) - D_i
                    trisk = [j for j in atrisk[t] if j not in failures[t]]
                    dt = len(failures[t]) * 1.0
                    s = np.sum(hz_ratio[trisk])
                    cj = 1 - dt / (dt + s)
                    s0.append(cj)
                else:
                    s0.append(1)
        elif algo == 'kp':
            for t in T[::-1]:
                if t in atrisk:
                    # R(t_i)
                    trisk = atrisk[t]
                    s = np.sum(hz_ratio[trisk])
                    si = hz_ratio[failures[t][0]]
                    cj = (1 - si / s) ** (1 / si)
                    s0.append(cj)
                else:
                    s0.append(1)
        elif algo == 'bsl':
            for t in T[::-1]:
                if t in atrisk:
                    # R(t_i)
                    trisk = atrisk[t]
                    dt = len(failures[t]) * 1.0
                    s = np.sum(hz_ratio[trisk])
                    cj = 1 - dt / s
                    s0.append(cj)
                else:
                    s0.append(1)
        else:
            pass
        S0 = np.cumprod(s0, axis=0)
        T0 = np.insert(T[::-1], 0, 0, axis=0)

        if smoothed:
            # smooth the baseline hazard
            ss = SuperSmoother()

            #Check duplication points
            ss.fit(T0, S0, dy=100)
            S0 = ss.predict(T0)

        return T0, S0