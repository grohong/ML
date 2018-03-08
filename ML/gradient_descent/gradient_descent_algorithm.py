import tensorflow
import matplotlib.pyplot as matplotlib

X = [1, 2, 3]
Y = [1, 2, 3]

W = tensorflow.placeholder(tensorflow.float32)

# 가정 H(x) = Wx
hypothesis = X * W

# cost(W) = 1/m 시그마 (Wx - y)^2
cost = tensorflow.reduce_mean(tensorflow.square(hypothesis - Y))

session = tensorflow.Session()
session.run(tensorflow.global_variables_initializer())

W_val = []
cost_val = []

for i in range(-30, 50):
    feed_W = i * 0.1
    current_cost, current_W = session.run([cost, W], feed_dict={W: feed_W})
    W_val.append(current_W)
    cost_val.append(current_cost)


matplotlib.plot(W_val, cost_val)
matplotlib.show()