import tensorflow
import numpy

x_data = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=numpy.float32)
y_data = numpy.array([[0], [1], [1], [0]], dtype=numpy.float32)

X = tensorflow.placeholder(tensorflow.float32)
Y = tensorflow.placeholder(tensorflow.float32)

W = tensorflow.Variable(tensorflow.random_normal([2, 1]), name='weight')
b = tensorflow.Variable(tensorflow.random_normal([1]), name='bias')

hypothesis = tensorflow.sigmoid(tensorflow.matmul(X, W) + b)

cost = -tensorflow.reduce_mean(Y*tensorflow.log(hypothesis) + (1-Y)*tensorflow.log(1-hypothesis))
train = tensorflow.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tensorflow.cast(hypothesis > 0.5, dtype=tensorflow.float32)
accuracy = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(predicted, Y), dtype=tensorflow.float32))

with tensorflow.Session() as session:
    session.run(tensorflow.global_variables_initializer())

    for step in range(10001):
        session.run(train, feed_dict={X:x_data, Y:y_data})

        if step%100 == 0:
            print(step, session.run(cost, feed_dict={X: x_data, Y: y_data}), session.run(W))

    h, c, a = session.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)