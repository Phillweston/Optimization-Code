import numpy as np
import cvxpy as cvx
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data[0:100]
y = iris.target[0:100]


X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.33)


n_samples, n_features = X_train.shape


def Sigmoid(z):

	G_of_Z = float(1.0 / float((1.0 + np.math.exp(-1.0*z))))

	return G_of_Z 


def Hypothesis(theta, x):

	z = 0
	for i in np.arange(len(theta)):
		z += x[i]*theta[i]

	return Sigmoid(z)

def Cost_Function(X,y,theta,m):

	sumOfErrors = 0
	for i in np.arange(m):
		xi = X[i]
		hi = Hypothesis(theta,xi)
		if y[i] == 1:
			error = y[i] * np.math.log(hi)
		elif y[i] == 0:
			error = (1-y[i]) * np.math.log(1-hi)
		sumOfErrors += error
	const = -1/m
	J = const * sumOfErrors

	print('cost is ', J)
	return J

def Cost_Function_Derivative(X,y,theta,j,m,alpha):

	sumErrors = 0
	for i in np.arange(m):
		xi = X[i]
		xij = xi[j]
		hi = Hypothesis(theta,X[i])
		error = (hi - y[i])*xij
		sumErrors += error

	m = len(y)
	constant = float(alpha)/float(m)
	J = constant * sumErrors
	return J


def Gradient_Descent(X,y,theta,m,alpha):

	new_theta = []
	for j in np.arange(len(theta)):
		CFDerivative = Cost_Function_Derivative(X,y,theta,j,m,alpha)
		new_theta_value = theta[j] - CFDerivative
		new_theta.append(new_theta_value)
	return new_theta

def Logistic_Regression(X,y,alpha,theta,num_iters):
    m = len(y)
    for x in np.arange(num_iters):
        new_theta = Gradient_Descent(X,y,theta,m,alpha)
        theta = new_theta
        print('theta:',theta)
        print('cost:',Cost_Function(X,y,theta,m))
    return theta


initial_theta = [0,0,0,0]
alpha = 0.01
iterations = 200
theta=Logistic_Regression(X_train,Y_train,alpha,initial_theta,iterations)
y_pred = [1 if Hypothesis(theta,x)>=0.5 else 0 for x in X_test ]
correct = np.mean([1 if x==y else 0 for (x,y) in zip(y_pred,Y_test)])
print(correct)
print(y_pred)
print(Y_test)