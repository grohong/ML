import tensorflow

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tensorflow.Variable(tensorflow.random_normal([1]), name='weight')
b = tensorflow.Variable(tensorflow.random_normal([1]), name='bias')

# Our hypothesis XW+b
hypothesis = x_train*W + b

#cost/Loss function
cost = tensorflow.reduce_mean(tensorflow.square(hypothesis - y_train))

# Minimize
optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


# Launch the graph in a session.
session = tensorflow.Session()
# Initializes global variables in the graph.
session.run(tensorflow.global_variables_initializer())

# Fit the line

for step in range(2001):
    session.run(train)
    if step % 20 == 0:
        print(step, session.run(cost), session.run(W), session.run(b))