import numpy as np
import scipy as sp
from scipy import integrate

#Integrating functions
x2 = lambda x: x**2
y,e = integrate.quad(x2,0,4)
print(y,e)

invexp = lambda x: np.exp(-x)
y,e =integrate.quad(invexp, 0, np.inf)
print(y,e)

f = lambda x,a : a*x
y, e = integrate.quad(f, 0, 1, args=(1,))
print(y,e)

y, e = integrate.quad(f, 0, 1, args=(3,))
print(y,e)

f = lambda x: 1 if x<=0 else 0
y, e = integrate.quad(f, -1, 1)
print(y,e)

f = lambda y, x: x*y**2
#Compute the double integral of x * y**2 over the box x ranging from 0 to 2 and y ranging from 0 to
#1
y, e = integrate.dblquad(f, 0, 2, lambda x: 0, lambda x: 1)
print(y,e)

#Compute the triple integral of x * y * z, over x ranging from 1 to 2, y ranging from 2 to 3, z ranging
#from 0 to 1.
f = lambda z, y, x: x*y*z
y, e = integrate.tplquad(f, 1, 2, lambda x: 2, lambda x: 3,lambda x, y: 0, lambda x, y: 1)
print(y,e)

#Linear algebra
from scipy import linalg
a = np.array([[1., 2.], [3., 4.]])
print(linalg.inv(a))

print(np.dot(a, linalg.inv(a)))

a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
b = np.array([2, 4, -1])
x = linalg.solve(a, b)
print(x)



