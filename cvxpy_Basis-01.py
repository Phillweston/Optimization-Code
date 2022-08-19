import numpy as np
import cvxpy as cvx

####################################################################
## 变量定义Variable和Parameter
##
####################################################################
#Parameters:	
#shape (tuple or int) – The leaf dimensions. Either an integer n for a 1D shape, or a tuple where the semantics are the same as NumPy ndarray shapes. Shapes cannot be more than 2D.
#value (numeric type) – A value to assign to the leaf.
#nonneg (bool) – Is the variable constrained to be nonnegative?
#nonpos (bool) – Is the variable constrained to be nonpositive?
#complex (bool) – Is the variable complex valued?
#symmetric (bool) – Is the variable symmetric?
#diag (bool) – Is the variable diagonal?
#PSD (bool) – Is the variable constrained to be positive semidefinite?
#NSD (bool) – Is the variable constrained to be negative semidefinite?
#Hermitian (bool) – Is the variable Hermitian?
#boolean (bool or list of tuple) – Is the variable boolean? True, which constrains the entire Variable to be boolean, False, or a list of indices which should be constrained as boolean, where each index is a tuple of length exactly equal to the length of shape.
#integer (bool or list of tuple) – Is the variable integer? The semantics are the same as the boolean argument.
#sparsity (list of tuplewith) – Fixed sparsity pattern for the variable.
#Method:.T,shape,ndim,size,value


# A scalar variable.
a = cvx.Variable()
# Vector variable with shape (5,).
x = cvx.Variable(5)
# Matrix variable with shape (5, 1).
x = cvx.Variable((5, 1))
# Matrix variable with shape (4, 7).
A = cvx.Variable((4, 7))

X = cvx.Variable((5,5),PSD=True)

x = cvx.Variable(2, complex=False)
y = cvx.Variable(2, complex=True)
z = cvx.Variable(2, imag=True)

a = cvx.Parameter(4,nonneg=True)
a.value = np.array([1,4,6,2],dtype=np.float)


####################################################################
## 运算规则
##
####################################################################
c = np.arange(5)
x = cvx.Variable(5)
print((c*x).shape) #缺省为內积
print(cvx.multiply(c,x).shape) #元素乘

c = np.arange(5).reshape((5,1))
x = cvx.Variable((5,1))
#print((c*x).shape) #error
print((c.T*x).shape) #ok
print(cvx.multiply(c,x).shape) 

c = np.arange(10).reshape((5,2))
x = cvx.Variable((5,2))
#print((c*x).shape) #error
print((c.T*x).shape) #true 
print(cvx.multiply(c,x).shape)


c = np.arange(5)
x = cvx.Variable((5,5))
print((c*x).shape) #true
#print(cvx.multiply(c,x).shape) #error
print((c.T*x).shape) #true
print((x*c).shape)
print((c*x*c).shape)
print((c.T*x*c).shape)


####################################################################
## 定义约束
##
####################################################################
x = cvx.Variable(5)
constraints = []
constraints.append(x >= 0)   #all x_i>=0
constraints.append(x <= 10)   
constraints.append(cvx.sum(x[:3]) <= 3)  # only part of vector; sum(first-three) <=3

for constr in constraints: #
    print(constr,constr.is_dcp())
#    print(constr,constr.curvature)
n=5
x = cvx.Variable((n,n))
A = np.random.rand(n,n)
constraints = []
constraints +=[(x >> 0)] #正半定约束    
constraints += [(cvx.trace(cvx.multiply(A,x))>=0) ] 



n=5
x = cvx.Variable((n,n))
A = np.random.rand(n,n)
constraints = []
constraints +=[(x >> 0),(cvx.trace(cvx.multiply(A,x))>=0)] #正半定约束    


x = cvx.Variable(2, name='x')
y = cvx.Variable(3, name='y')
z = cvx.Variable(2, name='z')
t = cvx.Variable()
exp = x + z
scalar_exp = 2.0+t
constr = cvx.SOC(scalar_exp, exp)#二阶锥约束 SOC ->cvx.norm(exp)<=scalar_exp
#constr= cvx.norm(exp)<=scalar_exp
constraints = []
constraints.append(constr) 

M = cvx.Variable((3, 3), PSD=True) #PSD变量
C1 = np.array([[0, 0, 1/2], [0, 0, 0], [1/2, 0, 1]])
C2 = np.array([[0, 0, 0], [0, 0, 1/2], [0, 1/2, 1]])
#x1 = cvx.Variable((3,3), PSD=True)
#x2 = cvx.Variable((3,3), PSD=True)
#constraints = [M + C1 == x1] #正半定约束
#constraints += [M + C2 == x2] #正半定约束
constraints = []
constraints.append(M + C1>>0)
constraints.append(M + C2>>0)
objective = cvx.Minimize(cvx.trace(M))
prob = cvx.Problem(objective, constraints)
prob.solve()
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", M.value)

####################################################################
## 组合新的矩阵
##
####################################################################
A = np.arange(6).reshape(3,2)
B = cvx.Variable((3,1))
C = cvx.Variable((1,2))
t = cvx.Variable()

D = cvx.bmat([[A,B],[C,cvx.reshape(t,(1,1))]])
print(D.shape)

E = cvx.vstack([cvx.hstack([A,B]),cvx.hstack([C,cvx.reshape(t,(1,1))])])
print(E.shape)



