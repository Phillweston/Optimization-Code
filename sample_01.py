import cvxpy as cvx

# Create 3 scalar optimization variables.
x = cvx.Variable(3)
x1 = x[0]
x2 = x[1]
x3 = x[2]
objective =-4*x1-5*x3

# Create two constraints.
constraints = [18*x1-x2+5*x3<=1,2*x1+4*x3<=1,10*x1+x2<=1,x<=1]
# Form objective.
obj = cvx.Minimize(objective)

# Form and solve problem.
prob = cvx.Problem(obj, constraints)
print(prob.is_dcp())
solvers=[cvx.SCS,cvx.CVXOPT,cvx.GUROBI,cvx.MOSEK]
for solver in solvers:
    prob.solve(solver=solver)  # Returns the optimal value.
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var", x.value)

#prob.solve() returns the optimal value and updates prob.status, prob.value, 
#and the value field of all the variables in the problem
  
       