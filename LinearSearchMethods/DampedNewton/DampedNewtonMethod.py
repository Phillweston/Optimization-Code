#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-21 22:41
# @Abstract：Damped Newton Method

'''
min_{x \in R^2} f(x)=100(x^2_1−x_2)^2+(x_1−1)^2
solution: x^*=(1,1)^T,f(x*)=0
grad: g(x)=(400*x_1(x^2_1−x_2)+2(x_1−1),−200(x^2_1−x_2))
'''

from numpy import *
from matplotlib.pyplot import *

def fun(x_k):
    return 100 * (x_k[0, 0] ** 2 - x_k[1, 0]) ** 2 + (x_k[0, 0] - 1) ** 2

# grad fun
def fun_grad(x_k):
    result = zeros((2, 1))
    result[0, 0] = 400 * x_k[0, 0] * (x_k[0, 0] ** 2 - x_k[1, 0]) + 2 * (x_k[0, 0] - 1)
    result[1, 0] = -200 * (x_k[0, 0] ** 2 - x_k[1, 0])
    return result

# grad fun_2
def fun_grad_2(x_k):
    result = zeros((2, 2))
    result[0, 0] = 1200 * x_k[0, 0]**2 - 400 * x_k[1, 0] + 2
    result[0, 1] = -400 * x_k[1, 0]
    result[1, 0] = -400 * x_k[1, 0]
    result[1, 1] = 200
    return result

def DampedNewton(x_k): # x0
    epsilon = 1e-4  # iteration condtion
    curve_y = [fun(x_k)]
    curve_x = [x_k]

    while abs(fun_grad(x_k)[0]) > epsilon:
        Hessian_k = fun_grad_2(x_k)
        Hessian_k_inverse = np.linalg.inv(Hessian_k)
        p_k = - Hessian_k_inverse * fun_grad(x_k)

