import numpy as np
import cvxpy as cvx

x = cvx.Variable()

g = 0.25*cvx.power(x-2,2)+4*x -17
f = cvx.sum_squares(x)
x.value =0.0  #给定初始点
for i in np.arange(20): 
    g_linear = cvx.linearize(g)
    obj = cvx.Minimize(f-g_linear)
    prob = cvx.Problem(obj)
    prob.solve(solver=cvx.CVXOPT)  # Returns the optimal value.
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var", x.value)


x = cvx.Variable(5)
constraints = []
constraints.append(x >= 0)   # all vars
constraints.append(x <= 10)  # all vars
constraints.append(cvx.sum(x[:3]) <= 3)  # only part of vector; sum(first-three) <=3
objective = cvx.Maximize(cvx.sum(x))
problem = cvx.Problem(objective, constraints)
problem.solve()
print(problem.status)
print(x.value.T)


c = np.arange(5)

print((c*x).shape)
print(cvx.multiply(c,x).shape)

c = np.arange(5).reshape((5,1))
x = cvx.Variable((5,1))

#print((c*x).shape) #error
print((c.T*x).shape) #true
print(cvx.multiply(c,x).shape)

c = np.arange(10).reshape((5,2))
x = cvx.Variable((5,2))

#print((c*x).shape) #error
print((c.T*x).shape) #true
print(cvx.multiply(c,x).shape)

