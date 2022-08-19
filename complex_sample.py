import cvxpy as cvx
import numpy as np
import scipy.linalg as la

#example 13.2 (d)
# Create two scalar optimization variables.
#x.T*H*x+2*P.T*x+r -->norm(H**(1/2)*x+H**(-1/2)*P)**2+r-P.T*inv(H)*P

#f(x)=x1^2+x2^2+0.5*x3^2+x1*x2+x1*x3-4*x1-3*x2-2*x3
#subject to:
#           -x1-x2-x3>=-3


K = np.matrix(np.random.rand(2,2) + 1j * np.random.rand(2,2) ) #  example matrix
n1 = la.svdvals(K).sum()  # trace norm of K

# Dual Problem
X = cvx.Variable((2,2), complex=True)
Y = cvx.Variable((2,2), complex=True)
# X, Y >= 0 so trace is real
objective = cvx.Minimize(
    cvx.real(0.5 * cvx.trace(X) + 0.5 * cvx.trace(Y))
)
constraints = [
    cvx.bmat([[X, -K.H], [-K, Y]]) >> 0,
    X >> 0,
    Y >> 0,
]
prob = cvx.Problem(objective, constraints)
result=prob.solve(solver=cvx.SCS,verbose=True)  # Returns the optimal value.

print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", X.value)
print("optimal var", Y.value)

