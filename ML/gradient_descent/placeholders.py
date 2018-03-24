import tensorflow

X = tensorflow.placeholder(tensorflow.float32)
Y = tensorflow.placeholder(tensorflow.float32)

W = tensorflow.Variable(tensorflow.random_normal([1]), name='weight')
b = tensorflow.Variable(tensorflow.random_normal([1]), name='bias')

# Our hypothesis XW+b
hypothesis = X*W + b

#cost/Loss function
cost = tensorflow.reduce_mean(tensorflow.square(hypothesis - Y))

# Minimize
optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


# Launch the graph in a session.
session = tensorflow.Session()
# Initializes global variables in the graph.
session.run(tensorflow.global_variables_initializer())

# Fit the line

for step in range(2001):
    cost_val, W_val, b_val, _ = \
        session.run([cost, W, b, train],
            feed_dict={X: [1, 2, 3],
                       Y: [1, 2, 3]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)