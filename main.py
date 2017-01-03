import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def Naive_Softmax():
    x = tf.placeholder('float', [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder('float', [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
    #cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for i in xrange(10):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

def CNN():
    def weigh_variable(shape):
        init = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init)

    def bias_variable(shape):
        init = tf.constant(0.1, shape=shape)
        return tf.Variable(init)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    x = tf.placeholder('float',shape=[None, 28*28])
    y_ = tf.placeholder('float', shape=[None, 10])

    W_conv1 = weigh_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    W_conv2 = weigh_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])


    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weigh_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    W_fc2 = weigh_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1, W_fc2)+b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    for i in xrange(500):
        batch = mnist.train.next_batch(50)
        #train_step.run(feed_dict={x: batch[0], y_:batch[1],keep_prob:0.5})
        sess.run(train_step,feed_dict={x: batch[0], y_:batch[1],keep_prob:0.5})
        if (i%50==0):
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_:batch[1],keep_prob:0.5})
            loss = sess.run(cross_entropy, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
            print 'train_accuracy ', train_accuracy, 'loss ', loss

CNN()

