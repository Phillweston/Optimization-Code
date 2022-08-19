# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 16:20:01 2018

@author: Administrator
"""

import cvxpy as cvx
from cvxpy.transforms.partial_optimize import partial_optimize
x = cvx.Variable()
t = cvx.Variable()
abs_x = partial_optimize(cvx.Problem(cvx.Minimize(sum(t)),
          [-t <= x, x <= t]), opt_vars=[t]).value



print(abs_x,t.value)