import numpy as np
import cvxpy as cvx


x = cvx.Variable()
print('Linearize...')
g = 0.25*cvx.sum_squares(x-2)+4*x -17
f = cvx.sum_squares(x)
x.value =0.01  #给定初始点
for i in np.arange(20): 
    g_linear = cvx.linearize(g)
    obj = cvx.Minimize(f-g_linear)
    prob = cvx.Problem(obj)
    prob.solve(solver=cvx.SCS)  # Returns the optimal value.
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var", x.value)




print('original...')
x = cvx.Variable()
f = 0.75*cvx.sum_squares(x)-3*x+16
obj = cvx.Minimize(f)
prob = cvx.Problem(obj)
prob.solve(solver=cvx.SCS)  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)