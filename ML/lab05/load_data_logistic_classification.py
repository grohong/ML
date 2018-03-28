import tensorflow
import numpy

xy = numpy.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=numpy.float32)
print("==========================")
print(xy)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tensorflow.placeholder(tensorflow.float32, shape=[None, 8])
Y = tensorflow.placeholder(tensorflow.float32, shape=[None, 1])

W = tensorflow.Variable(tensorflow.random_normal([8, 1]), name='weight')
b = tensorflow.Variable(tensorflow.random_normal([1]), name='bias')

hypothesis =tensorflow.sigmoid(tensorflow.matmul(X, W) + b)
cost = -tensorflow.reduce_mean(Y*tensorflow.log(hypothesis) + (1 - Y)*tensorflow.log(1-hypothesis))
train = tensorflow.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted =tensorflow.cast(hypothesis > 0.5, dtype=tensorflow.float32)
accuracy = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(predicted, Y), dtype=tensorflow.float32))

with tensorflow.Session() as session:
    session.run(tensorflow.global_variables_initializer())

    feed = {X: x_data, Y: y_data}
    for step in range(10001):
        session.run(train, feed_dict=feed)
        if step%200 == 0:
            print(step, session.run(cost, feed))

    h, c, a = session.run([hypothesis, predicted, accuracy], feed_dict=feed)
    print("\nHypothesisL", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
