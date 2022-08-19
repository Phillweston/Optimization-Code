# Solves a bounded least-squares problem.

import cvxpy as cvx
import numpy

# Problem data.
m = 10
n = 5
numpy.random.seed(1)
A = numpy.random.randn(m, n)
b = numpy.random.randn(m)

# Construct the problem.
x = cvx.Variable(n)
objective = cvx.Minimize(cvx.sum_squares(A*x - b))
constraints = [0 <= x, x <= 1]
prob = cvx.Problem(objective, constraints)

print("Optimal value", prob.solve())
print("Optimal var")
print(x.value) # A numpy ndarray.