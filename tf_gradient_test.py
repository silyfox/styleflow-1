import tensorflow as tf
import numpy as np

from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

N = 28
Nin = N*N
Nout = 10
batch_size = 100
NUM_BATCHES = 10

x = tf.placeholder(tf.float32, [None, Nin])
W = tf.Variable(tf.zeros([Nin, Nout]))
b = tf.Variable(tf.zeros([Nout]))

y = tf.nn.softmax(tf.matmul(x, W) + b) #predicted labels
y_ = tf.placeholder(tf.float32, [None, Nout]) #correct labels

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for n in range(NUM_BATCHES):
	batch_xs, batch_ys = mnist.train.next_batch(batch_size)
	sess.run(train_step, feed_dict = {x : batch_xs, y_ : batch_ys})
	var_grad = tf.gradients(cross_entropy, [b])
	var_grad_val = sess.run(var_grad, feed_dict = {x : batch_xs, y_ : batch_ys})
	print sess.run(b)
	print np.array(var_grad_val[0])




