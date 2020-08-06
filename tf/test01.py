import tensorflow as tf
print(tf.__version__)

hello = tf.constant("hello Yellow")

# sess = tf.Session()
sess = tf.Session()

print(sess.run(hello))