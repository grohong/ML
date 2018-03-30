import tensorflow

x_data = [[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5], [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]

x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

X = tensorflow.placeholder("float", [None, 3])
Y = tensorflow.placeholder("float", [None, 3])

W = tensorflow.Variable(tensorflow.random_normal([3, 3]))
b = tensorflow.Variable(tensorflow.random_normal([3]))

hypothesis = tensorflow.nn.softmax(tensorflow.matmul(X, W)+b)
cost = tensorflow.reduce_mean(-tensorflow.reduce_sum(Y*tensorflow.log(hypothesis), axis=1))
optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=1e-10).minimize(cost)

prediction = tensorflow.arg_max(hypothesis, 1)
is_correct = tensorflow.equal(prediction, tensorflow.arg_max(Y, 1))
accuracy = tensorflow.reduce_mean(tensorflow.cast(is_correct, tensorflow.float32))

with tensorflow.Session() as session:
    session.run(tensorflow.global_variables_initializer())
    for step in range(201):
        cost_val, W_val, _ = session.run([cost, W, optimizer],
                                         feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, W_val)

    print("Prediction: ", session.run(prediction, feed_dict={X: x_test}))
    print("Accurcy: ", session.run(accuracy, feed_dict={X: x_test, Y: y_test}))