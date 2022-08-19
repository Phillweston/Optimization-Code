#coding=utf-8

'''
problem:
 mimimize (x-2)^2+(y-1)^2
 subject to
         x^2-y<=0
         x+y<=2
'''

import cvxpy as cvx

# create two scalar optimization Variable
x = cvx.Variable()
y = cvx.Variable()

# create two constraints
constraints=[ cvx.square(x)- y <= 0,
              x + y <= 2]

# optimization objective function
obj = cvx.Minimize(cvx.square(x-2)+cvx.square(y-1))
prob=cvx.Problem(obj,constraints)

# return the optimization value
prob.solve()
print("status:",prob.status)
print("optimization value:",prob.value)
print("optimization var:",x.value,y.value)

'''
status: optimal
optimization value: 0.9999999975731159
optimization var: 0.999999999955422 0.99999999920646
'''


