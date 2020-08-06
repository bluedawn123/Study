import tensorflow as tf
tf.set_random_seed(777)

x_train = [1,2,3]
y_train = [3,5,7]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')    #random_normal? 랜덤한 숫자 하나를 집어넣어 정규화
                                                         #1,2는 차원을 의미한다. 

sess = tf.Session()
sess.run(tf.global_variables_initializer())  #초기화를 시킨다음 sess.run을 해야한다. 
print(sess.run(W))

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  #이 이후로 모든 변수들이 초기화

    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])  #_ 는 결과를 보여주지 않는다. 

        if step % 20==0:
            print(step, cost_val, W_val, b_val)






