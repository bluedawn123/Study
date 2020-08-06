import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.datasets import mnist          
import numpy as np               
mnist.load_data()                                         

(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train = x_train.reshape(-1, 28*28).astype('float32')/255.
x_test = x_test.reshape(-1, 28*28).astype('float32')/255.

print(x_train.shape, x_test.shape)   # (60000, 28, 28)
print(y_train.shape, y_test.shape)   # (60000,)  


'''
x = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.float32, shape = [None, 10])
'''

learing_late = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train) / batch_size)      #60000 / 100

print(len(x_train))

x =  tf.compat.v1.placeholder(tf.float32, shape = [None, 784])
y =  tf.compat.v1.placeholder(tf.float32, shape = [None, 10])

keep_prob =  tf.compat.v1.placeholder(tf.float32)  #드롭아웃

#w1 = tf.Variable(tf.random_normal([784, 512]), name = 'weight' ) 아래와 같다. 
w1 = tf.compat.v1.get_variable(
                    "w1", shape = [784, 512],
                    initializer=tf.contrib.layers.xavier_initializer())
print("w1 : ", w1)
b1 = tf.compat.v1.Variable(tf.random.normal([512]))
print("b1 : ", b1)
L1 = tf.compat.v1.nn.selu(tf.compat.v1.matmul(x, w1) + b1)
print("L1 : ", L1)
L1 = tf.compat.v1.nn.dropout(L1, rate = 1 - keep_prob)
print("L1 : ", L1)
####################################첫번째 히든레이어 구성 ####################################


w2 = tf.get_variable("w2", shape = [512, 512], 
                    initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.selu(tf.matmul(L1, w2) + b2)
L2 = tf.nn.dropout(L2, keep_prob = keep_prob)


w3 = tf.get_variable("w3", shape = [512, 512], 
                    initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.selu(tf.matmul(L2, w3) + b3)
L3 = tf.nn.dropout(L3, keep_prob = keep_prob)


w4 = tf.get_variable("w4", shape = [512, 256], 
                    initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([256]))
L4 = tf.nn.selu(tf.matmul(L3, w4) + b4)
L4 = tf.nn.dropout(L4, keep_prob = keep_prob)


w5 = tf.get_variable("w4", shape = [256, 10], 
                    initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))


hypothesis = tf.nn.softmax(tf.matmul(L4, w5) + b5)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learing_late).minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())





for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(total_batch):
###################################################################################### 
        start = i * batch_size
        end = start + batch_size
        
        batch_xs, batch_ys = x_train[start : end], y_train[start:end]
        #batch_xs, batch_ys = x_train[0:100], y_train[0:100]     #100 = 배치사이즈
        #batch_xs, batch_ys = x_train[100:200], y_train[100:200]
        #batch_xs, batch_ys = x_train[200:300], y_train[200:300]

        #batch_xs, batch_ys = x_train[i:batch_size], y_train[i:batch_size]
        #batch_xs, batch_ys = x_train[i+batch_size:batch_size+batch_size], y_train[i:batch_size]


###################################################################################### 
       
        feed_dict = {x:batch_xs, y:batch_ys, keep_prob : 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch


    print('Epoch : ', '%04d' %(epoch + 1), 
          'cost = {:.9f}'.format(avg_cost))

print("훈련끝")

prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('acc : ', sess.run(accuracy,
                         feed_dict = {x:x_test, y:y_test,
                         keep_prob : 1}))
