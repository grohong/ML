import tensorflow

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

X = tensorflow.placeholder(tensorflow.float32, shape=[None, 2])
Y = tensorflow.placeholder(tensorflow.float32, shape=[None, 1])

W = tensorflow.Variable(tensorflow.random_normal([2, 1]), name='weight')
b = tensorflow.Variable(tensorflow.random_normal([1]), name='bias')

hypothesis = tensorflow.sigmoid(tensorflow.matmul(X, W) + b)

cost = -tensorflow.reduce_mean(Y*tensorflow.log(hypothesis) + (1-Y)*tensorflow.log(1-hypothesis))
train = tensorflow.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tensorflow.cast(hypothesis > 0.5, dtype=tensorflow.float32)
accuracy = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(predicted, Y), dtype=tensorflow.float32))

with tensorflow.Session() as session:
    session.run(tensorflow.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = session.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = session.run([hypothesis, predicted, accuracy],
                          feed_dict={X: x_data, Y: y_data})

    print("\nhypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
