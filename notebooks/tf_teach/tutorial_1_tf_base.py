# 导入TensorFlow
import tensorflow as tf
# 定义两个矩阵
x = tf.constant([[1.0, 2.0]])
w = tf.constant([[3.0], [4.0]])
# 矩阵乘法，得到计算图（Graph）
y = tf.matmul(x, w)
# 输出图
print(y)
# 使用Session计算图结果
with tf.Session() as sess:
    # 输出运行结果
    print(sess.run(y))