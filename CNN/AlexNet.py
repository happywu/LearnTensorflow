import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.preprocessing import MultiLabelBinarizer
from Dataprovider import Dataprovider

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def inference(image, Batch_size, NUM_CLASSES):

    #image = tf.image.resize_images(image, [227, 227])
    parameter = []
    with tf.name_scope('conv1') as scope:

        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96], stddev=1.0,
                                                 dtype=tf.float32, name='weights'))
        conv = tf.nn.conv2d(image, kernel, [1, 4, 4, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)

        tf.summary.histogram('conv1/kernel', kernel)

        parameter += [kernel, biases]

    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')

    with tf.name_scope('conv2') as scope:

        kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=1.0,
                                                 dtype=tf.float32, name='weights'))
        conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)

        tf.summary.histogram('conv2/kernel', kernel)
        parameter += [kernel, biases]

    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')



    with tf.name_scope('conv3') as scope:

        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=1.0,
                                                 dtype=tf.float32, name='weights'))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)

        parameter += [kernel, biases]

    with tf.name_scope('conv4') as scope:

        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=1.0,
                                                 dtype=tf.float32, name='weights'))
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)

        parameter += [kernel, biases]

    with tf.name_scope('conv5') as scope:

        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=1.0,
                                                 dtype=tf.float32, name='weights'))
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)

        parameter += [kernel, biases]


    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')

    with tf.name_scope('fc6') as scope:
        # if batch_size known, tf.reshape(pool5, [batch_size, -1])
        # then we don't need to calculate the size of the image before fc layer
        reshape = tf.reshape(pool5, [Batch_size, -1])
        #dim = reshape.get_shape()[1].value
        dim = 6*6*256
        weights = tf.Variable(tf.truncated_normal([dim, 4096], stddev=1e-1,
                                                  dtype=tf.float32, name='weights'))
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                             trainable=True, name='biases')
        fc6 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope)

        parameter += [weights, biases]

    with tf.name_scope('fc7') as scope:
        weights = tf.Variable(tf.truncated_normal([4096, 1024], stddev=1e-1,
                                                  dtype=tf.float32, name='weights'))
        biases = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
                             trainable=True, name='biases')
        fc7 = tf.nn.relu(tf.matmul(fc6, weights) + biases, name=scope)

        parameter += [weights, biases]

    with tf.name_scope('fc8') as scope:
        # if batch_size known, tf.reshape(pool5, [batch_size, -1])
        # then we don't need to calculate the size of the image before fc layer
        weights = tf.Variable(tf.truncated_normal([1024, NUM_CLASSES], stddev=1e-1,
                                                  dtype=tf.float32, name='weights'))
        biases = tf.Variable(tf.constant(0.0, shape=[NUM_CLASSES], dtype=tf.float32),
                             trainable=True, name='biases')
        fc8 = tf.nn.relu(tf.matmul(fc7, weights) + biases, name=scope)

        parameter += [weights, biases]

    return fc8, parameter


def time_tensorflow_run(sess, target, info_string, feed_dict):
    sess.run(target, feed_dict=feed_dict)


images = tf.placeholder('float',shape=[None, 227, 227, 3], name="input_x")
y = tf.placeholder('float', shape=[None, 29], name="input_y")

fc8, parameter = inference(images, 50, 29)
fc8_softmax = tf.nn.softmax(fc8)

with tf.name_scope('loss'):
    objective = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc8_softmax, y))
    tf.summary.scalar("objective", objective)

with tf.name_scope('grad'):
    grad = tf.gradients(objective, parameter)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.1).minimize(objective)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(fc8_softmax, 1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)

X, Y = Dataprovider.getdata('../data/Caltech-101', 1)
Y = np.array(Y)
Y = np.reshape(Y,[-1,1])
Y = MultiLabelBinarizer().fit_transform(Y)

for i in xrange(5):
    #batch = mnist.train.next_batch(50)

    #image = np.array(list(batch[0]))
    image = np.array(X[i*50:(i+1)*50])
    _y = np.array(Y[i*50:(i+1)*50])
    # print image.shape
    # print _y.shape

    #image = np.reshape(image, [-1, 28, 28, 1])
    #image = tf.image.resize_images(image, [227, 227])

    feed_dict = {
        images : image,
        y: _y
    }
    sess.run(train_step, feed_dict=feed_dict)
    accu = sess.run(accuracy, feed_dict=feed_dict)
    loss = sess.run(objective, feed_dict=feed_dict)
    print 'accu: ', accu, 'loss ', loss
    #print 'k1', sess.run(parameter[0], feed_dict=feed_dict)
    #print 'gradient', sess.run(grad, feed_dict=feed_dict)
    result = sess.run(merged, feed_dict=feed_dict)
    writer.add_summary(result, i)



















