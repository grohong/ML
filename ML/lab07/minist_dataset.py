from tensorflow.examples.tutorials.mnist import input_data
import tensorflow
import random
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

X = tensorflow.placeholder(tensorflow.float32, [None, 784])
Y = tensorflow.placeholder(tensorflow.float32, [None, nb_classes])

W = tensorflow.Variable(tensorflow.random_normal([784, nb_classes]))
b = tensorflow.Variable(tensorflow.random_normal([nb_classes]))

hypothesis = tensorflow.nn.softmax(tensorflow.matmul(X, W) + b)

cost = tensorflow.reduce_mean(-tensorflow.reduce_sum(Y * tensorflow.log(hypothesis), axis=1))
optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

is_correct = tensorflow.equal(tensorflow.arg_max(hypothesis, 1), tensorflow.arg_max(Y, 1))
accuracy = tensorflow.reduce_mean(tensorflow.cast(is_correct, tensorflow.float32))


training_epochs = 15
batch_size = 100

with tensorflow.Session() as session:
    session.run(tensorflow.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost=0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = session.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch: ', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")
    print("Accuracy: ", accuracy.eval(session=session, feed_dict={X: mnist.test.images, Y:mnist.test.labels}))


    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", session.run(tensorflow.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", session.run(
        tensorflow.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

    plt.imshow(
        mnist.test.images[r:r + 1].reshape(28, 28),
        cmap='Greys',
        interpolation='nearest')
    plt.show()