import tensorflow as tf
import numpy as np

a = np.random.rand(3,2,2)
_a = tf.transpose(a, [1,0,2])
print a
print '###'
_a = tf.reshape(_a, [-1, 2])
with tf.Session() as sess:
    print _a.eval()
