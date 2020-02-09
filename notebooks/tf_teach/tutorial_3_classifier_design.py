import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw(x_1, x_2, x_3):
    x1_range = np.linspace(-10, 10)
    x2_range = np.linspace(-10, 10)
    x3_range = np.linspace(-10, 10)
    y_1 = (5 * x1_range + 3 * x2_range - 1) ** 2
    y_2 = (-3 * x1_range - 4 * x3_range + 1) ** 2
    fig = plt.figure(figsize=(16, 12))
    ax = Axes3D(fig)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("classifier design", alpha=0.5)
    ax.plot(x1_range, x2_range, y_1, label='y_1')
    ax.plot(x1_range, x3_range, y_2, label='y_2')
    ax.scatter(x_1, x_2, (5 * x_1 + 3 * x_2 - 1) ** 2, label="y_1_minima", color="green")
    ax.scatter(x_1, x_3, (-3 * x_1 - 4 * x_3 + 1) ** 2, label="y_2_minima", color="red")
    ax.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    x_1 = tf.Variable(0, dtype=tf.float32)
    x_2 = tf.Variable(0, dtype=tf.float32)
    x_3 = tf.Variable(0, dtype=tf.float32)

    y_1 = tf.add(tf.add(tf.multiply(5., x_1), tf.multiply(3., x_2)), -1.)
    y_2 = tf.add(tf.add(tf.multiply(-3., x_1), tf.multiply(-4., x_3)), 1.)
    cost = tf.add(tf.square(y_1), tf.square(y_2))
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    init = tf.global_variables_initializer()
    x_1_res = tf.Variable(0, dtype=tf.float32)
    x_2_res = tf.Variable(0, dtype=tf.float32)
    x_3_res = tf.Variable(0, dtype=tf.float32)
    cost_res = tf.Variable(0, dtype=tf.float32)
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            sess.run(train)
            cost_res = sess.run(cost)
            x_1_res = sess.run(x_1)
            x_2_res = sess.run(x_2)
            x_3_res = sess.run(x_3)
    draw(x_1_res, x_2_res, x_3_res)





