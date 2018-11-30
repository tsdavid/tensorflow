# import tensorflow as tf
#
#
#
# x_train = [1,2,3]
# y_train = [1,2,3]
#
# W = tf.Variable(tf.random_normal([1]), name="weight")
# b = tf.Variable(tf.random_normal([1]) , name="bias")
#
# #Our hypothesis H(x) = WX + b
# hypothesis = x_train * W + b
#
# #cost fuct
# cost = tf.reduce_mean(tf.square(hypothesis - y_train))
#
# #Minimize
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)
#
# #Launch the graph in a session
# sess = tf.Session()
# #Initializes global variables in the graph
# sess.run(tf.global_variables_initializer())
#
# #Fit the line
# for step in range(2001):
#     sess.run(train)
#     if step % 20 == 0:
#         print(step , sess.run(cost) , sess.run(W) , sess.run(b) )

#hypothsis with placeholders

import tensorflow as tf
W = tf.Variable(tf.random_normal([1]) , name="weight")
b = tf.Variable(tf.random_normal([1]) , name="bias")
X = tf.placeholder(tf.float32 , shape=[None])
Y = tf.placeholder(tf.float32 , shape=[None])
sess = tf.Session()

#Our hypothesis XW+b
hypothesis = X * W + b
#Cost / Loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
#Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
#Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

#Fit the line with new training data.
for step in range(2001):
    cost_val , w_val , b_val , _ = sess.run([cost , W, b ,train] , feed_dict={X:[1,2,3,4,5] ,Y:[2.1,3.1,4.1,5.1,6.1]})

    if step % 20 == 0:
        print(step , cost_val,w_val,b_val)




