import tensorflow
import numpy

xy = numpy.loadtxt('data-03-diabetes.csv', delimiter=', ', dtype=numpy.float32)
print("==========================")
print(xy)
x_data = xy[:, 0:-1]
y_data = xy[:, -1]

