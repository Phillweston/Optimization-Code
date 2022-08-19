#cvxpy prob.status
import cvxpy as cvx

x = cvx.Variable()

# An infeasible problem.
prob = cvx.Problem(cvx.Minimize(x), [x >= 1, x <= 0])
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)

# An unbounded problem.
prob = cvx.Problem(cvx.Minimize(x))
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)
