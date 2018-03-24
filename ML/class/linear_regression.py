import tensorflow as tensorflow

# training data
x_data = [1]
y_data = [1]


# neuron
w = tensorflow.Variable(tensorflow.random_normal([1]))
hypo = w*x_data


# learning
cost = (hypo - y_data) ** 2

train = tensorflow.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

session = tensorflow.Session()
session.run(tensorflow.global_variables_initializer())

for i in range(1001):
    session.run(train)

    if i%100 == 0:
        print('w: ', session.run(w), 'cost: ', session.run(cost))

# testing(prediction)
x_data = [2]
print(session.run(x_data*w))