# encoding: UTF-8

# Thomas Zhang for SBP in ANN
# 2019-4-2

import tensorflow as tf
import tensorflowvisu
import math
import mnistdata
import ipdb

print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnistdata.read_data_sets("data", one_hot=True, reshape=False)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# variable learning rate
lr = tf.placeholder(tf.float32)
# train/test selector for batch normalisation
tst = tf.placeholder(tf.bool)
# training iteration
iter = tf.placeholder(tf.int32)

# five layers and their number of neurons (tha last layer has 10 softmax neurons)
L = 100  # 200
# M = 100
# N = 60
# P = 30
Q = 10

# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 784 = 28 * 28
B1 = tf.Variable(tf.ones([L])/10)
# W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
# B2 = tf.Variable(tf.ones([M])/10)
# W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
# B3 = tf.Variable(tf.ones([N])/10)
# W4 = tf.Variable(tf.truncated_normal([N, P], stddev=0.1))
# B4 = tf.Variable(tf.ones([P])/10)
W5 = tf.Variable(tf.truncated_normal([L, Q], stddev=0.1))
B5 = tf.Variable(tf.ones([Q])/10)

# K = tf.Variable(tf.ones([Q])/10)
# Theta = tf.Variable(tf.ones([Q])/10)

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages

def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()

# The model
XX = tf.reshape(X, [-1, 784])
Y1l = tf.matmul(XX, W1)
Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1)
Y1 = tf.nn.relu(Y1bn)

# Y2l = tf.matmul(Y1, W2)
# Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2)
# Y2 = tf.nn.relu(Y2bn)

# Y3l = tf.matmul(Y2, W3)
# Y3bn, update_ema3 = batchnorm(Y3l, tst, iter, B3)
# Y3 = tf.nn.relu(Y3bn)

# Y4l = tf.matmul(Y3, W4)
# Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4)
# Y4 = tf.nn.relu(Y4bn)

Ylogits = tf.matmul(Y1, W5) + B5
Y = tf.nn.softmax(Ylogits)

update_ema = tf.group(update_ema1)#, update_ema2, update_ema3, update_ema4)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

####################################################################################
# cross entropy + SBPs

# sbp_bp
# pre \detW  \propto post \detW
# sbp_bp = XX, Y1l, Y1, Ylogits, 
# tf.transpose(XX):(784,?), Y1l:(?,100)
# predetW : (784,100)

ukuj = tf.matmul(tf.transpose(XX),Y1l)
ujui = tf.matmul(tf.transpose(Y1),Ylogits)


cost_function = cross_entropy

## SBP added into the ANN thoams: 2019-4-1


# sbp_post



## end added SBP in ANN







# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# matplotlib visualisation
# allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1])], 0)
# allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1])], 0)
allweights = tf.concat([tf.reshape(W1, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1])], 0)

# to use for sigmoid
#allactivations = tf.concat([tf.reshape(Y1, [-1]), tf.reshape(Y2, [-1]), tf.reshape(Y3, [-1]), tf.reshape(Y4, [-1])], 0)
# # to use for RELU
# allactivations = tf.concat([tf.reduce_max(Y1, [0]), tf.reduce_max(Y2, [0]), tf.reduce_max(Y3, [0]), tf.reduce_max(Y4, [0])], 0)
# alllogits = tf.concat([tf.reshape(Y1l, [-1]), tf.reshape(Y2l, [-1]), tf.reshape(Y3l, [-1]), tf.reshape(Y4l, [-1])], 0)
allactivations = tf.concat([tf.reduce_max(Y1, [0])], 0)
alllogits = tf.concat([tf.reshape(Y1l, [-1])], 0)
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)
# datavis = tensorflowvisu.MnistDataVis(title4="Logits", title5="Max activations across batch", histogram4colornum=2, histogram5colornum=2)

# training step
# the learning rate is: # 0.0001 + 0.03 * (1/e)^(step/1000)), i.e. exponential decay from 0.03->0.0001
lr = 0.0001 +  tf.train.exponential_decay(0.03, iter, 1000, 1/math.e)
train_step = tf.train.AdamOptimizer(lr).minimize(cost_function)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# ipdb.set_trace()


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):
    batch_X, batch_Y = mnist.train.next_batch(100)
    if update_train_data:
        a, c, im, al, ac, l = sess.run([accuracy, cost_function, I, alllogits, allactivations, lr],
                                       feed_dict={X: batch_X, Y_: batch_Y, iter: i, tst: False})
        print(str(i) + ": trainingaccuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(l) + ")")
    if update_test_data:
        a, c, im = sess.run([accuracy, cost_function, It], {X: mnist.test.images, Y_: mnist.test.labels, tst: True})
        # print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
    print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
    # the backpropagation training step
    sess.run([train_step, update_ema], feed_dict={X: batch_X, Y_: batch_Y, tst: False, iter: i})

for i in range(1000+1): 
    # training_step(i, i % 100 == 0, i % 20 == 0)
    training_step(i, i % 100 == 0, i % 20 == 0)

# print("max test accuracy: " + str(datavis.get_max_test_accuracy()))
