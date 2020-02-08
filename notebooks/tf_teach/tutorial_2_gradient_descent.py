# import tensorflow as tf
import sympy
import matplotlib.pyplot as plt
from numpy import *


# 函数实体，目标函数 y=x^2
def obj_func(x):
    return x ** 2


# 目标梯度，梯度可理解为目标函数的导数，高等数学里面求导的结果能表示函数的变化趋势，x < 0代表下降,越小斜率越大; x > 0代表上升，越大也是斜率越大
def obj_grad(x_input):
    x = sympy.Symbol("x")
    # 对函数进行求导, 结果为 2 * x
    dify = sympy.diff(x ** 2, x)
    # 返回梯度计算的值
    return dify.subs("x", x_input)


# 画图显示函数与梯度
def draw(x,y):
    range = linspace(-10, 10)
    plt.figure()
    plt.title('Gradient Descent')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.plot(range, range * 2, label="Target Gradient")
    plt.plot(range, range ** 2, label="The Original Function")
    plt.scatter(x, y, label="Local Minima", color="red")
    plt.legend(loc='best')
    plt.show()


"""
grad: 梯度函数
x: x的初始值
max_iter: 最大迭代次数
learning_rate: 学习率
precision: 收敛精度
"""

def gradient_descent(x, max_iter, learning_rate, precision):
    for i in range(max_iter):
        grad_value = obj_grad(x)
        if abs(grad_value) < precision:
            break
        x = x - grad_value * learning_rate
        print("第", i, "次迭代x的值为:", x)
    print("得到局部最小值时为x=", x, ", y=", obj_func(x))
    draw(x, obj_func(x))

if __name__ == '__main__':
    gradient_descent(5, 100, 0.01, 0.0001)