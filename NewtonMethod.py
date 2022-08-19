#coding=utf-8
'''
Newton’s method.
given a starting point x ∈ domf, tolerance ǫ > 0.
repeat
    1. Compute the Newton step and decrement.
     xnt := −∇2f(x)−1∇f(x); λ^2 := ∇f(x)^T∇2f(x)^{−1}∇f(x).
    2. Stopping criterion. quit if λ^2/2 ≤ ǫ.
    3. Line search. Choose step size t by backtracking line search.
    4. Update. x := x + t * xnt.
'''
import matplotlib.pyplot as plt


import numpy as np

def f(x):
    return (x-3)*(x-3)

def f_grad(x):
    return 2 * (x - 3)

def f_Hessian(x):
    return 2


# search step size
# x0: start point
def BacktrackingLineSearch(f,g,x0):
    # init data 0 < c < 0.5 (typical:10^-4 0) < rho <= 1
    alpha = 1
    x = x0
    rho = 0.8
    c = 1e-4

    # Armijo condition
    while f( x + alpha * (-g(x)) ) > f(x) + c * alpha * g(x) * (-g(x)) :
        alpha *= rho

    return alpha

# Newton Method
def NewtonMethod(f,g,H,x0):
    curve_y = [f(x0)]
    curve_x = [x0]

#    lambda_squre = 1
    eps = 1e-4
    error =10
    while error > eps:
        # stepSize = WolfeLineSearch(x0) # Wolfe condition
        stepSize = BacktrackingLineSearch(f,g,x0) # Armijo condition
#        stepSize = 0.01
        y0 = f(x0)
        d=- (1/H(x0))*(g(x0))
        x0 = x0 + stepSize * d

        y1 = f(x0)
#        lambda_squre=(f(x0)) * (1/(H(x0))) * (f(x0))
        error = np.linalg.norm(y0-y1)
        curve_x.append(x0)
        curve_y.append(y1)

    plt.plot(curve_y, 'g*-')
    plt. plot(curve_x, 'r+-')
    plt.xlabel('iterations')
    plt.ylabel('objective function value')
    plt.legend(['backtracking line search algorithm','x*'])
    plt.show()

if __name__ == "__main__":
    NewtonMethod(f,f_grad,f_Hessian,x0=0)


