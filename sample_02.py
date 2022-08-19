import cvxpy as cvx
import numpy as np
# Create two scalar optimization variables.
x = cvx.Variable(shape=(3,1))
c = cvx.Parameter(shape=(3,1))
c.value = np.mat([[-4],[0],[-8]])

A = cvx.Parameter(shape=(3,3))
A.value = np.mat([[18,-1,5],[2,0,4],[10,1,0]])
b = cvx.Parameter(shape=(3,1))
b.value = np.mat([[1],[1],[1]])
objective =c.T*x

# Create two constraints.
constraints = [A*x<=b,x<=1]
# Form objective.
obj = cvx.Minimize(objective)

# Form and solve problem.
prob = cvx.Problem(obj, constraints)
prob.solve(verbose=True)  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)

#prob.solve() returns the optimal value and updates prob.status, prob.value, 
#and the value field of all the variables in the problem
  
       