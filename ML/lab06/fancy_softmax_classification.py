import tensorflow
import numpy

xy = numpy.loadtxt('data-04-zoo.csv', delimiter=',', dtype=numpy.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7

X = tensorflow.placeholder(tensorflow.float32, [None, 16])
Y = tensorflow.placeholder(tensorflow.int32, [None, 1])

Y_one_hot = tensorflow.one_hot(Y, nb_classes)
Y_one_hot = tensorflow.reshape(Y_one_hot, [-1, nb_classes])

W = tensorflow.Variable(tensorflow.random_normal([16, nb_classes]), name='weight')
b = tensorflow.Variable(tensorflow.random_normal([nb_classes]), name='bias')

logits = tensorflow.matmul(X, W) + b
hypothesis = tensorflow.nn.softmax(logits)

cost_i = tensorflow.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                         labels=Y_one_hot)

cost = tensorflow.reduce_mean(cost_i)
optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tensorflow.argmax(hypothesis, 1)
correct_prediction = tensorflow.equal(prediction, tensorflow.argmax(Y_one_hot, 1))
accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))

with tensorflow.Session() as session:
    session.run(tensorflow.global_variables_initializer())

    for step in range(2001):
        session.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step%100 == 0:
            loss, acc = session.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc:{:.2%}".format(step, loss, acc))

    pred = session.run(prediction, feed_dict={X: x_data})
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

# # Lab 6 Softmax Classifier
# import tensorflow as tf
# import numpy as np
# tf.set_random_seed(777)  # for reproducibility
#
# # Predicting animal type based on various features
# xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
# x_data = xy[:, 0:-1]
# y_data = xy[:, [-1]]
#
# print(x_data.shape, y_data.shape)
#
# nb_classes = 7  # 0 ~ 6
#
# X = tf.placeholder(tf.float32, [None, 16])
# Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6
# Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
# print("one_hot", Y_one_hot)
# Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
# print("reshape", Y_one_hot)
#
# W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
# b = tf.Variable(tf.random_normal([nb_classes]), name='bias')
#
# # tf.nn.softmax computes softmax activations
# # softmax = exp(logits) / reduce_sum(exp(logits), dim)
# logits = tf.matmul(X, W) + b
# hypothesis = tf.nn.softmax(logits)
#
# # Cross entropy cost/loss
# cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
#                                                  labels=Y_one_hot)
# cost = tf.reduce_mean(cost_i)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
#
# prediction = tf.argmax(hypothesis, 1)
# correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# # Launch graph
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for step in range(2000):
#         sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
#         if step % 100 == 0:
#             loss, acc = sess.run([cost, accuracy], feed_dict={
#                                  X: x_data, Y: y_data})
#             print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
#                 step, loss, acc))
#
#     # Let's see if we can predict
#     pred = sess.run(prediction, feed_dict={X: x_data})
#     # y_data: (N,1) = flatten => (N, ) matches pred.shape
#     for p, y in zip(pred, y_data.flatten()):
#         print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))