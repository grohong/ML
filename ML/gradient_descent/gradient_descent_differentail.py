import tensorflow

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tensorflow.Variable(tensorflow.random_normal([1]), name='weight')
X = tensorflow.placeholder(tensorflow.float32)
Y = tensorflow.placeholder(tensorflow.float32)


hypothesis = X * W