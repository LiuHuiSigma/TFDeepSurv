### L1 or L2 regularizer

func:
	tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
links:
	http://blog.csdn.net/marsjhao/article/details/72831021

### Dropout and Activation

func:
	tf.nn.dropout(Wx_plus_b,keep_prob=0.5)
    activation_function=tf.nn.relu
links:
	https://www.cnblogs.com/lovephysics/p/7220574.html

### Adam optimizer

func:
	tf.train.AdamOptimizer(1e-4).minimize(loss)
	tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name=’Adam’)
links:
	

### learning rates decay

func:
	global_steps = tf.Variable(0, trainable=False)  
	learning_rate = tf.train.exponential_decay(0.1, global_steps, 10, 2, staircase=False) 

links:
	http://blog.csdn.net/uestc_c2_403/article/details/72403833

### Batch Norm

func:
	tf.nn.moments
	tf.nn.batch_normalization

links:
	http://blog.csdn.net/fontthrone/article/details/76652772