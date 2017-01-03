import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def inference(X, batch_size):
    x_image = tf.reshape(X, [-1, 28, 28, 1])
    parameter = []
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1,
                                                 dtype=tf.float32, name='weights'))
        conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)

        tf.summary.histogram('conv1/kernel', kernel)

        parameter += [kernel, biases]

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool1')

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, name='weights'))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)

        tf.summary.histogram('conv2/kernel', kernel)

        parameter += [kernel, biases]

    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool2')

    with tf.name_scope('fc1') as scope:
        reshape = tf.reshape(pool2, [batch_size, -1])
        weights = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=1e-1,
                                                  dtype=tf.float32, name='weights'))
        biases = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
                             trainable=True, name='biases')
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope)

        parameter += [weights, biases]

    with tf.name_scope('fc2') as scope:
        weights = tf.Variable(tf.truncated_normal([1024, 10], stddev=1e-1,
                                                  dtype=tf.float32, name='weights'))
        biases = tf.Variable(tf.constant(0.0, shape=[10], dtype=tf.float32),
                             trainable=True, name='biases')
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope)

        parameter += [weights, biases]

    return fc2, parameter

x = tf.placeholder('float',shape=[None, 28*28])
y_ = tf.placeholder('float', shape=[None, 10])


fc2, parameter = inference(x, 50)
fc2_softmax = tf.nn.softmax(fc2)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2_softmax, y_))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(fc2_softmax, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
keep_prob = tf.placeholder(tf.float32)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
for i in xrange(500):
    batch = mnist.train.next_batch(50)
    #train_step.run(feed_dict={x: batch[0], y_:batch[1],keep_prob:0.5})
    sess.run(train_step,feed_dict={x: batch[0], y_:batch[1],keep_prob:0.5})
    if (i%50==0):
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_:batch[1],keep_prob:0.5})
        loss = sess.run(cross_entropy, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
        print 'train_accuracy ', train_accuracy, 'loss ', loss
