import tensorflow
import numpy

x_data = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=numpy.float32)
y_data = numpy.array([[0], [1], [1], [0]], dtype=numpy.float32)

X = tensorflow.placeholder(tensorflow.float32)
Y = tensorflow.placeholder(tensorflow.float32)

W1 = tensorflow.Variable(tensorflow.random_normal([2, 2]), name='weight1')
b1 = tensorflow.Variable(tensorflow.random_normal([2]), name='bias1')
layer1 = tensorflow.sigmoid(tensorflow.matmul(X, W1) + b1)

W2 = tensorflow.Variable(tensorflow.random_normal([2, 1]), name='weight2')
b2 = tensorflow.Variable(tensorflow.random_normal([1]), name='bias2')
hypothesis = tensorflow.sigmoid(tensorflow.matmul(layer1, W2) + b2)

cost = -tensorflow.reduce_mean(Y*tensorflow.log(hypothesis) + (1-Y)*tensorflow.log(1-hypothesis))
train = tensorflow.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tensorflow.cast(hypothesis > 0.5, dtype=tensorflow.float32)
accuracy = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(predicted, Y), dtype=tensorflow.float32))

with tensorflow.Session() as session:
    session.run(tensorflow.global_variables_initializer())

    for step in range(10001):
        session.run(train, feed_dict={X:x_data, Y:y_data})

        if step%100 == 0:
            print(step, session.run(cost, feed_dict={X: x_data, Y: y_data}), session.run([W1, W2]))

    h, c, a = session.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: \n", h, "\nCorrect: \n", c, "\nAccuracy: ", a)