import tensorflow as tf

x_data = [[1,2] , [2,3] , [3,1] , [4,3] , [5,3] , [6,2]]
y_data = [[0] , [0] , [0] , [1] , [1] , [1]]


#placeholders for a tensor that will be always fed.

X = tf.placeholder(tf.float32 , shape = [None , 2])
Y = tf.placeholder(tf.float32 , shape = [None , 1])
W = tf.Variable(tf.random_normal([2,1]) , name="weight")
b = tf.Variable(tf.random_normal([1]) , name="bias")

#Hypothesis using sigmoid: tf.div(1. , 1. + tf.exp(tf.matmul(X,W) + b))
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)

#cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y)*tf.log(1 - hypothesis))

#Gradient Descent Function
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#Accuracy computation
#True if hyphothesis > 0/5 else False
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))

#Launch graph
with tf.Session() as sess:
    #Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    # for step in range(10001):
    #     cost_val , _ = sess.run([cost , train] , feed_dict={X :x_data , Y:y_data})
    #     if step % 200 == 0:
    #         print(step , cost_val)


    #Accuracy report
    # h,c,a = sess.run([hypothesis,predicted,accuracy],
    #                  feed_dict={X:x_data,Y:y_data})
    # #print("\nHypothesis: " ,h,"\nCorrect: " ,c, "\nAccuracy: ",a)


#당뇨병
import numpy as np
import tensorflow as tf
xy = np.loadtxt('data-03-diabetes.csv' , delimiter=',' , dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

#placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32 , shape=[None , 8])
Y = tf.placeholder(tf.float32 , shape=[None , 1])

W = tf.Variable(tf.random_normal([8,1]) , name="Weight")
b = tf.Variable(tf.random_normal([1]) , name="bias")

#Hyppthesis using sigmoid
hypothosis = tf.sigmoid(tf.matmul(X,W)+b)
#cost/loss function
cost = -tf.reduce_mean(Y*tf.log(hypothosis) + (1-Y)*tf.log(1-hypothosis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#Accuracy computation
predicted = tf.cast(hypothosis > 0.5 ,dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted , Y) ,dtype=tf.float32))
# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed = {X: x_data, Y: y_data}
    for step in range(10001):
        sess.run(train, feed_dict=feed)
        if step % 400 == 0:
            print(step, sess.run(cost, feed_dict=feed))

    # Accuracy report
    h, c, a = sess.run([hypothosis, predicted, accuracy], feed_dict=feed)
    print("\n:Hypothesis :", h, "\nCorrect: " ,c, "\nAccuracy: " , a )