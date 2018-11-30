# # import tensorflow as tf
# # import matplotlib.pyplot as plt
# #
# # X = [1,2,3]
# # Y = [1,2,3]
# #
# # W = tf.placeholder(tf.float32)
# # # Our hypothesis for linear model X*W
# # hypothesis = X*W
# #
# # #Cost Loss function
# # cost = tf.reduce_mean(tf.square(hypothesis - Y))
# # #Launch the graph in a session.
# # sess = tf.Session()
# # #Initializes global varialbes in the graph
# # sess.run(tf.global_variables_initializer())
# # #Variables for plotting cos function
# # W_val = []
# # cost_val = []
# # for i in range(-30 , 50):
# #     feed_W = i * 0.1
# #     curr_cost , curr_W = sess.run([cost , W] , feed_dict={W:feed_W})
# #     W_val.append(curr_W)
# #     cost_val.append(curr_cost)
# #
# # #show the cost function
# # plt.plot(W_val , cost_val)
# # plt.show()
# #
#
#
# import tensorflow as tf
#
# x_data = [1,2,3]
# y_data = [1,2,3]
#
# W = tf.Variable(tf.random_normal([1]) , name="weight")
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
#
# #Our hypothesis for linear model X*W
# hypothesis = X * Y
#
# #cost / loss function
# cost = tf.reduce_sum(tf.square(hypothesis-Y))
#
# #Minimize: Gradient Descent using derivative: W -= Learning_rate * derivate
# learning_rate = 0.1
# gradient = tf.reduce_mean((W*X-Y)*X)
# descent = W - learning_rate*gradient
# update = W.assign(descent)
#
# #Launch the graph in a session.
# sess = tf.Session()
# #Initializes global variables in the grah.
# sess.run(tf.global_variables_initializer())
# for step in range(21):
#     sess.run(update , feed_dict={X:x_data , Y:y_data})
#     print(step, sess.run(cost , feed_dict={X : x_data , Y:y_data}), sess.run(W))



#Adanced
import tensorflow as tf

X = [1,2,3]
Y = [1,2,3]
W = tf.Variable(5.)
#Our hypothesis for linear model X*W
hypothesis = X * W
#Manual gradient
gradient = tf.reduce_mean((W*X-Y)*X)*2
#cost / loss function
cost = tf.reduce_sum(tf.square(hypothesis-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

#Get gradients
gvs = optimizer.compute_gradients(cost)
#Apply gradients
apply_gradients = optimizer.apply_gradients(gvs)

#Launch the graph in a session.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step , sess.run([gradient , W , gvs]))
    sess.run(apply_gradients)
