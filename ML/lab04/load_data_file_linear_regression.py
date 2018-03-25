import tensorflow
import numpy

xy = numpy.loadtxt('data-01-test-score.csv', delimiter=",", dtype=numpy.float32)
# print(xy)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# print(x_data.shape, x_data, len(x_data))
# print(y_data.shape, y_data)

X = tensorflow.placeholder(tensorflow.float32, shape=[None, 3])
Y = tensorflow.placeholder(tensorflow.float32, shape=[None, 1])

W = tensorflow.Variable(tensorflow.random_normal([3, 1]), name='weight')
b = tensorflow.Variable(tensorflow.random_normal([1]), name='bias')

hypothesis = tensorflow.matmul(X, W) + b

cost = tensorflow.reduce_mean(tensorflow.square(hypothesis-Y))

optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

session = tensorflow.Session()
session.run(tensorflow.global_variables_initializer())

for step in range(2001):
    cost_val, hypothesis_val, _ = session.run(
        [cost, hypothesis, train],
        feed_dict={X: x_data, Y: y_data})

    if step%10 == 0:
        print(step, "Cost: ", cost_val,
              "\nPrediction: \n", hypothesis_val)

print("Your score will be ", session.run(hypothesis,
                                         feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be", session.run(hypothesis,
                                          feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))