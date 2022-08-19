import cvxpy as cvx
import numpy as np

#Linear Programming-->SDP
c=np.array([-1,-4],dtype=np.float64);
A=np.array([[-1, 0],[-1 ,-1],[-1 ,-2]],dtype=np.float64);
b=np.array([[-2],[ -3.5],[ -6]],dtype=np.float64);
C=np.diag(c);

n=2;
x = cvx.Variable(shape=(2,1))
X= cvx.diag(x)
objective =cvx.trace(C.T*X)
obj = cvx.Minimize(objective)
constraints = [cvx.diag(A*x-b)>>0]
constraints += [X>>0]


# Form and solve problem.
prob = cvx.Problem(obj, constraints)
prob.solve(solver=cvx.CVXOPT)  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)



#SDP problem
M = cvx.Variable((3, 3), PSD=True)
C1 = np.array([[0, 0, 1/2], [0, 0, 0], [1/2, 0, 1]])
C2 = np.array([[0, 0, 0], [0, 0, 1/2], [0, 1/2, 1]])

#constraints = [M + C1 == cvx.Variable((3,3), PSD=True)]
#constraints += [M + C2 == cvx.Variable((3,3), PSD=True)]
constraints = [M + C1 >>0]
constraints += [M + C2 >>0]
objective = cvx.Minimize(cvx.trace(M))
prob = cvx.Problem(objective, constraints)
opt_val = prob.solve(solver=cvx.CVXOPT)
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)

#QP-->SDP
H=np.array([[1,-0.5],[-0.5,1]],dtype=np.float64)
P0=np.array([[-3],[0]],dtype=np.float64)
H_inv = np.linalg.inv(H)
A=np.array([[-1,-1],[-2,1]],dtype=np.float64)
b=np.array([[-2],[-2]],dtype=np.float64)
x = cvx.Variable((2, 1))
t = cvx.Variable()
X = cvx.vstack([cvx.hstack([H_inv,x]),cvx.hstack([x.T,t-P0.T*x])])
constraints = [X== cvx.Variable((3,3), PSD=True)]
constraints += [A*x>=b]
objective = cvx.Minimize(t)
prob = cvx.Problem(objective, constraints)
prob.solve(solver=cvx.CVXOPT)
#opt_val = prob.solve(solver=cvx.SCS)
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)

#QCQP -->SDP
Q0= np.array([[1,0,-1,0],[0,1,0,-1],[-1,0,1,0],[0,-1,0,1]])
D,G=np.linalg.eig(Q0);
Q00=G.dot(np.diag(np.sqrt(D))).T
Q1=np.array([[1/4,0],[0,1]])
D,G=np.linalg.eig(Q1);
Q11=G.dot(np.diag(np.sqrt(D))).T
Q2=np.array([[5/8,3/8],[3/8,5/8]])
D,G=np.linalg.eig(Q2);
Q22=G.dot(np.diag(np.sqrt(D))).T
P1=np.array([[-1/2],[0]])
P2=np.array([[-11/2],[-13/2]])
b1=3/4
b2=-35/2
c0 = np.array([[0],[0],[0],[0]])
x = cvx.Variable((4, 1))
t = cvx.Variable()
x1 = x[0]
x2 = x[1]
x3 = x[2]
x4 = x[3]

y = cvx.vstack([x1,x2])
z = cvx.vstack([x3,x4])
#X1 = cvx.vstack([cvx.hstack([np.eye(4),Q00*x]),cvx.hstack([(Q00*x).T,t - c0.T*x])])
X1 = cvx.vstack([cvx.hstack([np.eye(4),Q00*x]),cvx.hstack([(Q00*x).T,cvx.reshape(t,(1,1))])])
X2 = cvx.vstack([cvx.hstack([np.eye(2),Q11*y]),cvx.hstack([(Q11*y).T,b1-P1.T*y])])
X3 = cvx.vstack([cvx.hstack([np.eye(2),Q22*z]),cvx.hstack([(Q22*z).T,b2-P2.T*z])])
constraints = [X1>>0,X2>>0,X3>>0]
objective = cvx.Minimize(t)
prob = cvx.Problem(objective, constraints)
prob.solve(solver=cvx.CVXOPT)
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)


#QCQP -->SOCP
Q0= np.array([[1,0,-1,0],[0,1,0,-1],[-1,0,1,0],[0,-1,0,1]])
D,G=np.linalg.eig(Q0);
Q00=G.dot(np.diag(np.sqrt(D))).T
Q1=np.array([[1/4,0],[0,1]])
D,G=np.linalg.eig(Q1);
Q11=G.dot(np.diag(np.sqrt(D))).T
Q2=np.array([[5/8,3/8],[3/8,5/8]])
D,G=np.linalg.eig(Q2);
Q22=G.dot(np.diag(np.sqrt(D))).T
P1=np.array([[-1/2],[0]])
P2=np.array([[-11/2],[-13/2]])
b1=3/4
b2=-35/2
#c0 = np.array([[0],[0],[0],[0]])
x = cvx.Variable((4, 1))
t = cvx.Variable()
x1 = x[0]
x2 = x[1]
x3 = x[2]
x4 = x[3]

y = cvx.vstack([x1,x2])
z = cvx.vstack([x3,x4])

#constraints = [cvx.norm(Q00*x)<=cvx.sqrt(t - c0.T*x)]
#constraints = [cvx.norm(Q00*x)<=cvx.sqrt(t)]
#constraints += [cvx.norm(Q11*y)<=cvx.sqrt(b1-P1.T*y)]
#constraints += [cvx.norm(Q22*z)<=cvx.sqrt(b2-P2.T*z)]

constraints = []
constraints.append(cvx.norm(Q00*x)<=cvx.sqrt(t - c0.T*x))   # all vars
constraints.append(cvx.norm(Q11*y)<=cvx.sqrt(b1-P1.T*y)) 
constraints.append(cvx.norm(Q22*z)<=cvx.sqrt(b2-P2.T*z)) 

constraints += [b1-P1.T*y>=0]
constraints += [b2-P2.T*z>=0]
objective = cvx.Minimize(t)
prob = cvx.Problem(objective, constraints)
prob.solve(solver=cvx.CVXOPT)
#prob.solve(solver=cvx.ECOS)
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value)




