#cvxpy DCP
import cvxpy as cvx
x = cvx.Variable()
y = cvx.Variable()

# DCP problems.
prob1 = cvx.Problem(cvx.Minimize(cvx.square(x - y)),
                    [x + y >= 0])
prob2 = cvx.Problem(cvx.Maximize(cvx.sqrt(x - y)),
                [2*x - 3 == y,
                 cvx.square(x) <= 2])
print("prob1 is DCP:", prob1.is_dcp())
print("prob2 is DCP:", prob2.is_dcp())
# Non-DCP problems.
# A non-DCP objective.
obj = cvx.Maximize(cvx.square(x))
prob3 = cvx.Problem(obj)
print("prob3 is DCP:", prob3.is_dcp())
print("Maximize(square(x)) is DCP:", obj.is_dcp())

# A non-DCP constraint.
constraints= [cvx.sqrt(x) <= 2]
prob4 = cvx.Problem(cvx.Minimize(cvx.square(x)),
                    )

print ("prob4 is DCP:", prob4.is_dcp())
print ("cvx.square(x):", cvx.square(x).curvature)
print ("sqrt(x) <= 2 is DCP:", constraints[0].is_dcp())


