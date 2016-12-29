import tensorflow as tf
import numpy as np

tf.reset_default_graph()

# Create input data
X = np.random.randn(2, 10, 8)

# The second example is of length 6
X[1,6:] = 0
X_lengths = [10, 6]

cell = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True)

outputs, states = tf.nn.dynamic_rnn(
    inputs=X,
    cell=cell,
    dtype=tf.float64,
    sequence_length=X_lengths)

result = tf.contrib.learn.run_n(
    {'outputs':outputs},
    n=1,
    feed_dict=None)


print len(result)
print result[0]['outputs'].shape
print result[0]

