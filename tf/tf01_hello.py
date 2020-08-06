import tensorflow as tf
print(tf.__version__)

hello = tf.constant("hello world")  #constant = 상수
print(hello)
#Tensor("Const:0", shape=(), dtype=string)

sess = tf.Session()
print(sess.run(hello))  


