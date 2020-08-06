# 3 + 4 + 5
# 4 - 3
# 3 * 4
# 4 / 2
import tensorflow as tf

node2 = tf.constant(2.0, tf.float32)
node3 = tf.constant(3.0, tf.float32)
node4 = tf.constant(4.0)
node5 = tf.constant(5.0)

d = tf.multiply(node3, node4)  # a * b
e = tf.add(node3, node4, node5)  # c + b
f = tf.subtract(node4, node3)  # d - e





