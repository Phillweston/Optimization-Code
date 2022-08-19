import cvxpy as cvx
import numpy as np

#Variable(shape, boolean=True) instead of Bool(shape).
#Variable(shape, integer=True) instead of Int(shape).
#Variable((n, n), PSD=True) instead of Semidef(n).
#Variable((n, n), symmetric=True) instead of Symmetric(n).
#Variable(shape, nonneg=True) instead of NonNegative(shape).
#Parameter(shape, nonneg=True) instead of Parameter(shape, sign='positive').
#Parameter(shape, nonpos=True) instead of Parameter(shape, sign='negative').

#Parameters:	
#shape (tuple or int) – The variable dimensions (0D by default). Cannot be more than 2D.
#name (str) – The variable name.
#value (numeric type) – A value to assign to the variable.
#nonneg (bool) – Is the variable constrained to be nonnegative?
#nonpos (bool) – Is the variable constrained to be nonpositive?
#symmetric (bool) – Is the variable constrained to be symmetric?
#hermitian (bool) – Is the variable constrained to be Hermitian?
#diag (bool) – Is the variable constrained to be diagonal?
#complex (bool) – Is the variable complex valued?
#imag (bool) – Is the variable purely imaginary?
#PSD (bool) – Is the variable constrained to be symmetric positive semidefinite?
#NSD (bool) – Is the variable constrained to be symmetric negative semidefinite?
#boolean (bool or list of tuple) – Is the variable boolean (i.e., 0 or 1)? True, which constrains the entire variable to be boolean, False, or a list of indices which should be constrained as boolean, where each index is a tuple of length exactly equal to the length of shape.
#integer (bool or list of tuple) – Is the variable integer? The semantics are the same as the boolean argument.

#11.11
# Create two scalar optimization variables.
x = cvx.Variable(shape=(3,1))

c = np.array([[2],[-6],[3]],dtype=np.float64)

A = np.array([[-3,1,-2],[2,-4,0],[4,-3,-3]],dtype=np.float64)
b = np.array([[-7],[-12],[-14]],dtype=np.float64)
objective =c.T*x

# Create two constraints.
constraints = [A*x>=b,x>=0]
# Form objective.
obj = cvx.Minimize(objective)

# Form and solve problem.
prob = cvx.Problem(obj, constraints)
prob.solve(verbose=True)  # Returns the optimal value.
print('11.16 result:')
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)

##11.16
# Create two scalar optimization variables.
x = cvx.Variable(shape=(4,1))

c = np.array([[1],[1.5],[1],[1]],dtype=np.float64)


A = np.array([[1,2,1,2],[1,1,2,4]],dtype=np.float64)

b = np.array([[3],[5]],dtype=np.float64)
objective =c.T*x

# Create two constraints.
constraints = [A*x==b,x>=0]
# Form objective.
obj = cvx.Minimize(objective)

# Form and solve problem.
prob = cvx.Problem(obj, constraints)
prob.solve(verbose=True)  # Returns the optimal value.
print('11.16 result:')
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)



#11.17
# Create two scalar optimization variables.
x = cvx.Variable(shape=(3,1))
c = np.array([[1],[0.5],[2]],dtype=np.float64)


A = np.array([[1,1,2],[2,1,3]],dtype=np.float64)

b = np.array([[3],[5]],dtype=np.float64)
objective =c.T*x

# Create two constraints.
constraints = [A*x==b,x>=0]
# Form objective.
obj = cvx.Minimize(objective)

# Form and solve problem.
prob = cvx.Problem(obj, constraints)
prob.solve(verbose=True)  # Returns the optimal value.
print('11.17 result:')
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)

#prob.solve() returns the optimal value and updates prob.status, prob.value, 
#and the value field of all the variables in the problem
  
       