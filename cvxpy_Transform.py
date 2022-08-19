import numpy as np
import cvxpy as cvx
from cvxpy.transforms.indicator import indicator
x = cvx.Variable()
constraints = [0 <= x, x <= 1]
expr = indicator(constraints)
x.value = .5
print("expr.value = ", expr.value)
x.value = 2
print("expr.value = ", expr.value)


