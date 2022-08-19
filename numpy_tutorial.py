import numpy as np

print(np.__version__)
#datatype
#np.int8,np.int16,np.int32,np.int64
#np.uint8,np.uint16,np.uint32,np.uint64
#np.float16,np.float32,np.float64
#np.complex64,np.complex128

#constant
#np.inf,np.NAN,np.NZERO,np.PZERO,np.e,np.nan,np.newaxis
#np.pi
x = np.arange(12).reshape(3,4)
print(x.ndim)
print(x.shape)
print(x.dtype)
print(x.size)
y=x.astype(np.float32)
print(y.dtype)

#numpy create
l=[1,2,3,4]
x = np.array(l)
print(x)
x = np.asarray(l)
print(x)

x=np.array([[1,2,3],[4,5,6]])
print(x)

x = np.arange(start=0,stop=10,step=2)
print(x)

x= np.linspace(start=0,stop=10,num=20)
print(x)
print(x.shape)

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
x= np.zeros((2,3),dtype=np.int32)
print(x)

x= np.ones((4,3))
print(x)

y = np.zeros_like(x)
print(y)

z= np.empty_like(y)
print(z)

x = np.random.rand(3,4)
print(x)

x = np.arange(12).reshape(2,6)
print(x)
print(x.shape)
y = x.reshape(-1,2) #-1自动计算长度
print(y)
x.resize((4,3))
print(x.shape)
print(x)

print(x.flatten())
print(x.flatten(order='C')) #缺省，按行flatten
print(x.flatten(order='F')) #按列flatten

#Copying/Sorting
y = x.copy()
y[1]=0
print(y)

x = np.random.rand(3,4)
y = x.copy()
print(x)
x.sort() #每行排序
print(x)

y.sort(axis=0) #每列排序
print(y)



#Combining Arrays
x=np.array([1,2,3])
y=np.array([4,5,6])
z = np.concatenate((x,y)) #一维只有一种组合方式
print(z)
print(z.shape)
x=x[np.newaxis,:] #增加长度为1的新维度
print(x)
print(x.shape)
y=y[np.newaxis,:]
print(y)
print(y.shape)
z = np.concatenate((x,y))
print(z)
print(z.shape)
z = np.concatenate((x,y),axis=1) 
print(z)
print(z.shape)

x=np.array([1,2,3])
y=np.array([4,5,6])
x=x[:,np.newaxis]
print(x)
print(x.shape)
y=y[:,np.newaxis]
print(y)
print(y.shape)
z = np.concatenate((x,y)) 
print(z)
print(z.shape)
z = np.concatenate((x,y),axis=1) 
print(z)
print(z.shape)

x=np.array([[1,2,3],[4,5,6]])
print(x)
print(x.shape)
y=np.array([[7,8,9],[10,11,12]])
print(y)
print(y.shape)
z = np.concatenate((x,y),axis=0) 
print(z)
print(z.shape)
z = np.concatenate((x,y),axis=1) 
print(z)
print(z.shape)

z=np.vstack((x,y))
print(z)
print(z.shape)

z=np.hstack((x,y))
print(z)
print(z.shape)

x=np.array([[1,2,3]]) #x,y维度不一致
print(x)
print(x.shape)
y=np.array([[7,8,9],[10,11,12]])
print(y)
print(y.shape)
z = np.concatenate((x,y),axis=0) 
print(z)
print(z.shape)

z=np.vstack((x,y))
print(z)
print(z.shape)

x = np.array((1,2,3))
y = np.array((4,5,6))
print(np.column_stack((x, y)))

#Splitting Arrays
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print(np.array_split(x, 3)) #拆分为3个
x = np.arange(9)
print(np.split(x, 3))
x = np.random.rand(20,4)
print(np.split(x,[3,5,12,19],axis=0)) #指定拆分indices

#repeat and title
x = [1,2,3]
y = np.repeat(x,2)
print(y)
y = np.tile(x,(2,3))
print(y)

x = [[1,2,3],[4,5,6]]
y = np.repeat(x,2)
print(y)
y = np.tile(x,(2,3))
print(y)
#More
x = np.arange(12).reshape(4,3)
print(x)
y=x.flatten()
print(y)
print(y.shape)

#Transpose
x = np.random.rand(3,4)
print(x)
print(x.transpose())
x = np.random.rand(3,2,4)
y = np.transpose(x,(1,0,2))
print(y.shape)

z = np.swapaxes(x,0,1)
print(z.shape)

x = [[3,1],[2,4]]
print(np.linalg.inv(x))

#Mathematics elemwise
x = np.random.rand(3,2)
y = np.ones_like(x)
print(np.add(x,y))

print(np.subtract(x,y))
print(np.divide(x,y))
print(np.multiply(x,y)) #element-wise multiplication
print(np.sqrt(x))
print(np.dot(x,y.T)) #matrix multiplication

#Comparison
x = np.random.rand(3,2)
indices = x>=0.5 #==,!=,>=,<=,>,<
print(indices)
x[1,1]=0
x[1,:]=0
nz = np.nonzero(x)
print(nz)

Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1
print(Z)


#Basic Statistics
x = np.array([1, 1, 2, 5, 8, 10, 11, 12])
print(np.median(x))
print(np.mean(x))
print(x.var())
print(x.std())

x = np.random.rand(4,3)
print(np.median(x))
print(np.mean(x))
print(x.var())
print(x.std())

print(np.sum(x,axis=0))
print(np.sum(x,axis=1))
print(np.sum(x))

print(np.max(x,axis=0))  #min
print(np.max(x,axis=1))
print(np.max(x))

x = np.array(
    [[1, 2, 3, 4],
     [5, 6, 7, 8]])

outs = [np.sum(x),
        np.sum(x, axis=0),
        np.sum(x, axis=1, keepdims=True),
        "",
        np.prod(x),
        np.prod(x, axis=0),
        np.prod(x, axis=1, keepdims=True),
        "",
        np.cumsum(x),
        np.cumsum(x, axis=0),
        np.cumsum(x, axis=1),
        "",
        np.cumprod(x),
        np.cumprod(x, axis=0),
        np.cumprod(x, axis=1),
        "",
        np.min(x),
        np.min(x, axis=0),
        np.min(x, axis=1, keepdims=True),
        "",
        np.max(x),
        np.max(x, axis=0),
        np.max(x, axis=1, keepdims=True),
        "",
        np.mean(x),
        np.mean(x, axis=0),
        np.mean(x, axis=1, keepdims=True)]
           
for out in outs:
    if out == "":
        print("")
    else:
        print("->", out)

#expand/squeeze dim
x = np.zeros((3,2),dtype=np.int32)
y = np.expand_dims(x,axis=0)
print(y.shape)
z= x[np.newaxis,...]
print(z.shape)

w = np.squeeze(z)
print(w.shape)

#count unique
x = np.array([2, 2, 1, 5, 4, 5, 1, 2, 3])
u, indices = np.unique(x, return_counts=True)
print (u, indices)

#Slicing and Subsetting
#a = np.random.rand(10)
a = np.arange(10)
print(a[np.array([3, 3, 1, 8])])
print(a)
print(a[3:5])  # 用范围作为下标获取数组的一个切片，包括a[3]不包括a[5]

print(a[:5])   # 省略开始下标，表示从a[0]开始

print(a[:-1])  # 下标可以使用负数，表示从数组后往前数

a[2:4] = 100,101    # 下标还可以用来修改元素的值
print(a)
print(a[1:-1:2])   # 范围中的第三个参数表示步长，2表示隔一个元素取一个元素

print(a[::-1]) # 省略范围的开始下标和结束下标，步长为-1，整个数组头尾颠倒

print(a[5:1:-2]) # 步长为负数时，开始下标必须大于结束下标

x = np.random.rand(10,10)
print(x[[1,3,5]][:,[5,8,9]]) #row=[1,4,5] col=[5,8,9]
print(x[np.ix_([1,3,5],[5,8,9])])
print(x[np.array([1,3,5]),np.array([5,8,9])]) #取[1,5],[3,8],[5,9]元素
print(x[1,5],x[3,8],x[5,9]) #取[1,5],[3,8],[5,9]元素
print(x[ ::2,:5])
print(x[ ::-1,:5])




#Logic
x = np.array([1,2,3])
print (np.all(x))

x = np.array([1,0,3])
print (np.all(x))

x = np.array([1,2,3])
print (np.any(x))

x = np.array([1,0,3])
print (np.any(x))

x = np.array([1, 0, np.nan, np.inf])
print (np.isfinite(x))

x = np.array([1, 0, np.nan, np.inf])
print (np.isinf(x))

x = np.array([1, 0, np.nan, np.inf])
print (np.isnan(x))

x = np.array([1+1j, 1+0j, 4.5, 3, 2, 2j])
print (np.iscomplex(x))
x = np.array([1+1j, 1+0j, 4.5, 3, 2, 2j])
print (np.isreal(x))

x = np.array([4, 5])
y = np.array([2, 5])
indices=np.greater(x, y)
print (x[indices])
print (np.greater_equal(x, y))
print (np.less(x, y))
print (np.less_equal(x, y))

print (np.logical_and([True, False], [False, False]))
print (np.logical_or([True, False, True], [True, False, False]))
print (np.logical_xor([True, False, True], [True, False, False]))
print (np.logical_not([True, False, 0, 1]))

#Random Sampling
#rand,randn,randint
x = np.random.rand(3, 2) 
print(x)
x = np.random.uniform(low=0,high=10,size=(3,4))
print(x)
out1 = np.random.randn(1000, 1000) 

m=2.0
sigma = 2.6
x = np.sqrt(sigma) * np.random.randn(200, 40) + m
print(np.var(x)) #sigma
outs = np.random.normal(loc=0.5, scale=2.0, size=(1000, 1000))
print(np.mean(outs),np.var(outs))

out=np.random.randint(0, 4, size=(3, 2))
print(out)
out = np.random.randint(2, size=10) #not including 2
print(out)

print(np.random.random_sample((5,)))
print(np.random.random((5,)))
#Three-by-two array of random numbers from [-5, 0):
#(b - a) * random_sample() + a [a=-5,b=0)
print(5 * np.random.random_sample((3, 2)) - 5) 
print(5 * np.random.sample((3, 2)) - 5)

x = [6, 2, 1]
for _ in range(10):
    print (np.random.choice(x, p=[.3, .5, .2]))
x = np.random.choice(10, 3, replace=False) #10 choose 3
print(x)
y = np.random.choice(10, size=(4,3), replace=True) #10 choose 3
print(y)

np.random.seed(1027)
x = np.arange(10)
np.random.shuffle(x)
print (x)
indices = np.random.permutation(10)
print(x[indices])

#Distributions
print(np.random.chisquare(2,4))
dfnum = 1. # between group degrees of freedom
dfden = 48. # within groups degrees of freedom
s = np.random.f(dfnum, dfden, 1000)

shape, scale = 2., 2. # mean=4, std=2*sqrt(2)
s = np.random.gamma(shape, scale, 1000)

loc, scale = 0., 1.
s = np.random.laplace(loc, scale, 1000)

mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)

s = np.random.uniform(-1,0,1000)
#Set routines
x = np.array([1, 2, 6, 4, 2, 3, 2])
out, indices = np.unique(x, return_inverse=True)
print(out,indices)

test = np.array([0, 1, 2, 5, 0])
states = [0, 2]
mask = np.in1d(test, states)
print(test[mask])
print(mask)

x = np.array([0, 1, 2, 5, 0])
y = np.array([0, 1, 4])
print (np.intersect1d(x, y))

x = np.array([0, 1, 2, 5, 0])
y = np.array([0, 1, 4])
print (np.setdiff1d(x, y))

x = np.array([0, 1, 2, 5, 0])
y = np.array([0, 1, 4])
out1 = np.union1d(x, y)
out2 = np.sort(np.unique(np.concatenate((x, y))))
print(out1)

#Soring, searching, and counting
x = np.random.rand(5,4)
print(x)
out = np.sort(x, axis=1)
x.sort(axis=1)
print(out)

x = np.random.rand(5,4)
print(x)
x.sort(axis=0)
print(x)

x = np.array([[1,4],[3,1]])
out = np.argsort(x, axis=1)
print (out)

x = np.random.permutation(10)
print ("x =", x)
out = np.partition(x, 5) #前5个小于5，后面大于5
print(out)
x = np.random.permutation(10)
print ("x =", x)
partitioned = np.partition(x, 3)
indices = np.argpartition(x, 3)
print( "partitioned =", partitioned)
print ("indices =", partitioned)
assert np.array_equiv(x[indices], partitioned)

x = np.random.permutation(10).reshape(2, 5)
print ("x =", x)
print ("maximum values =", np.max(x, axis=1))
print ("max indices =", np.argmax(x, axis=1))
print ("minimum values =", np.min(x, axis=1))
print ("min indices =", np.argmin(x, axis=1))

x = np.array([[1, 2, 3], [1, 3, 5]])
print ("Values bigger than 2 =", x[x>2])
print ("Their indices are ", np.nonzero(x > 2))

x = np.array([1, 2, 3,-2,5])
print ("Values bigger than 2 =", x[x>2])
print ("Their indices are ", np.nonzero(x > 2))

x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
print(np.nonzero(x)) #返回非0数的indices

x = np.arange(-5, 4).reshape(3, 3)
print(x)
print (np.where(x <0, 0, x**2))

#Statistics
x = np.arange(4).reshape((2, 2))
print("x=\n", x)
print("ans=\n", np.amax(x, axis=1, keepdims=True))

x = np.arange(10).reshape((2, 5))
print("x=\n", x)
out1 = np.ptp(x, 1) #peak_to_peak
out2 = np.amax(x, 1) - np.amin(x, 1)
assert np.allclose(out1, out2)
print("ans=\n", out1)

x = np.arange(1, 11).reshape((2, 5))
print("x=\n", x)
print("ans=\n", np.percentile(x, 75, axis=1)) #75%

x = np.arange(1, 10).reshape((3, 3))
print("x=\n", x)
print("ans=\n", np.median(x,axis=1))
print("ans=\n", np.median(x))

x = np.arange(5)
weights = np.arange(1, 6)
out1 = np.average(x, weights=weights)
out2 = (x*(weights/weights.sum())).sum()
assert np.allclose(out1, out2)
print(out1)

x = np.array([0, 1, 2])
y = np.array([2, 1, 0])
print("ans=\n", np.cov(x, y))

x = np.array([0, 1, 3])
y = np.array([2, 4, 5])
print("ans=\n", np.correlate(x, y))

#Matrix
x = np.matrix(np.arange(12).reshape((3,4)))
print(x)
y= x[1]
print(y.shape) #also matrix

x = np.arange(12).reshape((3,4))
y= x[1]
print(y.shape) #vector,not matrix

print(np.matrix([[1, 2], [3, 4]]))
print(np.matrix('1 2; 3 4'))

x = np.random.rand(3,2)
print(x)
print(x.flatten().tolist()) #转list

A = np.matrix(x)
print(A.getA()) #z换numpy ndarray
print(A.getA1().tolist()) #flatten转list

x = np.matrix(np.arange(12).reshape((3,4)))
print(x.argmax())
print(x.argmax(axis=0)) #matrix
print(x.argmax(axis=1)) #matrix

A = np.matrix([[2.0,1+1j],[1-2j,1j]])
B = np.matrix([[1.0,-1j],[1+1j,0]])
print(A)
print(A.conj())
print(A.imag)
print(A.real)
print(np.multiply(A,B)) #elem_multipy
print(A*B) #matrix multipy
print(A.H) #the conjugate transpose

import numpy.matlib as mt
A = mt.zeros((2,3))
print(A)
A = np.matlib.eye(3, k=1, dtype=float)
A = np.matlib.identity(3, dtype=int)
B = np.asmatrix(np.arange(6).reshape(2, 3))
A = np.matlib.rand(2, 3)
#randn


#Linear algebra
x = [1,2]
y = [[4, 1], [2, 2]]
print (np.dot(x, y)) #矩阵乘
print (np.dot(y, x))
print (np.matmul(x, y))
print (np.inner(x, y)) #np.tensordot
print (np.inner(y, x))

a = np.array([1,2,3])
b = np.array([0,1,0])
print(np.inner(a, b))

a = np.arange(24).reshape((2,3,4))
b = np.arange(4)
print(np.inner(a, b))

print(np.inner(np.eye(2), 7))
x = [[1, 0], [0, 1]]
y = [[4, 1], [2, 2], [1, 1]]
print (np.dot(y, x))
x = np.array([[1, 4], [5, 6]])
y = np.array([[4, 1], [2, 2]])
print (np.vdot(x, y)) #转向量内积
print (np.vdot(y, x))
print (np.dot(x.flatten(), y.flatten()))
print (np.inner(x.flatten(), y.flatten()))

x=np.kron(np.eye(2), np.ones((2,2)))
print(x)

x = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=np.int32)
L = np.linalg.cholesky(x)
print (L)
x = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]], dtype=np.float32)
q, r = np.linalg.qr(x)
print ("q=\n", q, "\nr=\n", r)
x = np.array([[1, 0, 0, 0, 2], [0, 0, 3, 0, 0], [0, 0, 0, 0, 0], [0, 2, 0, 0, 0]], dtype=np.float32)
U, s, V = np.linalg.svd(x, full_matrices=False)
print ("U=\n", U, "\ns=\n", s, "\nV=\n", V)

x = np.diag((1, 2, 3))
eigenvals,eigenvecs = np.linalg.eig(x)
print(eigenvals)
eigenvals_ = np.linalg.eigvals(x)
print ("eigenvectors are\n", eigenvecs)

x = np.arange(1, 10).reshape((3, 3))
print (np.linalg.norm(x, 'fro'))
print (np.linalg.cond(x, 'fro'))

a = np.arange(9) - 4
print(a)
b = a.reshape((3, 3))
print(b)
print(np.linalg.norm(a))
print(np.linalg.norm(b))
print(np.linalg.norm(b,'fro'))
print(np.linalg.norm(a,np.inf))
print(np.linalg.norm(b,np.inf))
print(np.linalg.norm(b,-np.inf))
print(np.linalg.norm(a,1))
print(np.linalg.norm(b,ord=-2,axis=1))
print(np.linalg.norm(b,ord=1,axis=0))

x = np.eye(4)
out1 = np.linalg.matrix_rank(x)
print(out1)

x = np.eye(4)
out1 = np.trace(x)
print(out1)

x = np.array([[1., 2.], [3., 4.]])
out1 = np.linalg.inv(x)
print(out1)

a = np.array([[3,1], [1,2]])
b = np.array([9,8])
x = np.linalg.solve(a, b)
print(x)


#Indexing routines
#np.c_,np.r_按列或按行拼接
print(np.c_[np.array([1,2,3]), np.array([4,5,6])])
print(np.r_[np.array([1,2,3]), np.array([4,5,6])])

x = np.random.randint(3,size=(4,5))
print(x)
indices = np.nonzero(x)
print(indices)
print(x[indices])
x = np.array([[1,2,3],[4,5,6],[7,8,9]])
indices = np.nonzero(x>3)
print(x[indices])

x = np.arange(9.).reshape(3, 3)
indices=np.where( x > 5 )
print(x[indices])
y=np.where(x < 5, x, -1)
print(y)

goodvalues = [3, 4, 7]
ix = np.isin(x, goodvalues)
print(np.where(ix)) #np.nonzero(ix)

a = np.arange(10).reshape(2, 5)
print(a)
ixgrid = np.ix_([0, 1], [2, 4]) #[0,2],[0,4],[1,2],[1,4] meshgrid
print(a[ixgrid])

di = np.diag_indices(4) #所有对角元素位置,3则只取前3个对角
a = np.arange(16).reshape(4, 4)
print(a)
a[di]=100
print(a)

iu = np.mask_indices(4, np.triu)
a[iu]=-2
print(a)

a = [4, 3, 5, 7, 6, 8]
indices = [0, 1, 4]
print(np.take(a,indices))

#Functional programming
def my_func(a):
       """Average first and last element of a 1-D array"""
       return (a[0] + a[-1]) * 0.5
b = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(np.apply_along_axis(my_func, 0, b)) #apply func to col
print(np.apply_along_axis(my_func, 1, b)) #apply func to row


#numpy  Search
np.lookfor('linspace')
np.source('linspace')
