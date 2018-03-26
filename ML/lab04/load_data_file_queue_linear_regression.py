import tensorflow

filename_queue = tensorflow.train.string_input_producer(
    ['data-01-test-score.csv'], shuffle=False, name='filename-queue'
)

reader = tensorflow.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[0.], [0.], [0.], [0.]]
xy = tensorflow.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = tensorflow.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

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

coordinator = tensorflow.train.Coordinator()
threads = tensorflow.train.start_queue_runners(sess=session, coord=coordinator)

for step in range(2001):
    x_batch, y_batch = session.run([train_x_batch, train_y_batch])
    print(x_batch)
    print("================================")
    print(y_batch)
    cost_val, hy_val, _ = session.run(
        [cost, hypothesis, train],
        feed_dict={X: x_batch, Y: y_batch}
    )

    if step % 10 == 0:
        print(step, "Cost: ", cost_val,
              "\nPrediction:\n", hy_val)

coordinator.request_stop
coordinator.join(threads)