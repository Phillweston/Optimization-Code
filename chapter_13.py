import cvxpy as cvx
import numpy as np

#example 13.2 (d)
# Create two scalar optimization variables.
#x.T*H*x+2*P.T*x+r -->norm(H**(1/2)*x+H**(-1/2)*P)**2+r-P.T*inv(H)*P

#f(x)=x1^2+x2^2+0.5*x3^2+x1*x2+x1*x3-4*x1-3*x2-2*x3
#subject to:
#           -x1-x2-x3>=-3


x = cvx.Variable(shape=(3,1))

H= np.array([[1.0000 , 0.5000, 0.5000],[ 0.5 , 1 ,0],[0.5 , 0 , 0.5]],dtype=np.float64) #H=L'*L

p = np.array([[-2],[-3/2],[-1]],dtype=np.float64)
A = np.array([[-1],[-1],[-1]],dtype=np.float64)
b = np.array([[-3]],dtype=np.float64)

objective =cvx.quad_form(x,H)+2*p.T*x

# Create two constraints.
constraints = [A.T*x>=b]
# Form objective.
obj = cvx.Minimize(objective)

# Form and solve problem.
prob = cvx.Problem(obj, constraints)
prob.solve(solver=cvx.SCS,verbose=True)  # Returns the optimal value.

print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)



np.random.seed(0)
A = np.random.randn(5, 5)
z = np.random.randn(5)
P = A.T.dot(A)
q = -2*P.dot(z)
w = cvx.Variable(5, name='w')

prob = cvx.Problem(cvx.Minimize(cvx.QuadForm(w, P) + q.T*w))
qp_solution = prob.solve(solver=cvx.SCS)
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)


A = np.random.randn(10, 5)
b = np.random.randn(10)
prob = cvx.Problem(cvx.Minimize(cvx.norm(A*w - b, 2)))
qp_solution = prob.solve(solver=cvx.CVXOPT)
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)

