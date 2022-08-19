#cvxpy Parameter
import cvxpy as cvx
import numpy
import matplotlib.pyplot as plt

# Problem data.
n = 15
m = 10
numpy.random.seed(1)
A = numpy.random.randn(n, m)
b = numpy.random.randn(n)
# gamma must be nonnegative due to DCP rules.
gamma = cvx.Parameter(nonneg=True)

# Construct the problem.
x = cvx.Variable(m)
error = cvx.sum_squares(A*x - b)
obj = cvx.Minimize(error + gamma*cvx.norm(x, 1))
prob = cvx.Problem(obj)

# Construct a trade-off curve of ||Ax-b||^2 vs. ||x||_1
sq_penalty = []
l1_penalty = []
x_values = []
gamma_vals = numpy.logspace(-4, 6)
for val in gamma_vals:
    gamma.value = val  #must assign value
    prob.solve()
    # Use expr.value to get the numerical value of
    # an expression in the problem.
    sq_penalty.append(error.value)
    l1_penalty.append(cvx.norm(x, 1).value)
    x_values.append(x.value)

print(sq_penalty)
print(l1_penalty)
print(x_values)
print('end')
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt.figure(figsize=(6,10))
#
## Plot trade-off curve.
#plt.subplot(211)
#plt.plot(l1_penalty, sq_penalty)
#plt.xlabel(r'\|x\|_1', fontsize=16)
#plt.ylabel(r'\|Ax-b\|^2', fontsize=16)
#plt.title('Trade-Off Curve for LASSO', fontsize=16)
#
## Plot entries of x vs. gamma.
#plt.subplot(212)
#for i in range(m):
#    plt.plot(gamma_vals, [xi[i] for xi in x_values])
#plt.xlabel(r'\gamma', fontsize=16)
#plt.ylabel(r'x_{i}', fontsize=16)
#plt.xscale('log')
#plt.title(r'\text{Entries of x vs. }\gamma', fontsize=16)
#
#plt.tight_layout()
#plt.show()