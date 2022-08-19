# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 06:07:42 2018

@author: Administrator
"""

from sympy import *
import numpy as np

x, y, z = symbols('x y z')
f=(exp(x)/factorial(y))**z
g = Subs(f,x,2)
pprint(g)

x, y = symbols('x, y', real=True)
f = cos(x) * exp(y)
print(N(f.subs({x: pi, y: 2})))

x = Symbol('x')
a = Symbol('a')
f=lambdify([x, a], a + x**2, "numpy") 
arr = np.random.randn(1000)
result = f(arr, 0.4)
print(result)

print(diff(exp(x**2), x))
print(diff(x**2 * y**2, y))
print(diff(x**3, x, x))
print(diff(x**3, x, 2))

g =y**2*exp(x**2)
print(diff(diff(g,x),y))


