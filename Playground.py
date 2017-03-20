"""----------------------------------------------
-----------------Matt Bowyer---------------------
----------------------------------------------"""

import math
import sympy
import numpy

#a = 0
#n = 5
#x = sympy.symbols('x', real=True)
#func_1 = sympy.exp(x) * sympy.sin(x)
#func_2 = sympy.log(1+x) - sympy.log(1-x)

def taylor(f, a, n):
    derivatives = [sympy.diff(f,x,i).subs(x,a) for i in range(n)]
    factors = [ math.factorial(i) for i in range(n)]
    distances = [ (x-a)**i for i in range(n)
    expresions = [distances[i]*derivatives[i]/factors[i] for i in range(n)]
    taylor_poly = sum(expressions)
    taylor_polynomial = sum([((x-a)**i) * sympy.diff(f,x,i).subs(x,a) /math.factorial(i) for i in range(n)] )

# a=1
# itr=10
# x = sympy.symbols('x')
# func_1 = x**2 - 2
# func_2 = 1 - sympy.exp(x)
# func_3 = x**2 - sympy.sin(x) - 0.5

def newtonsMethod(f, a, itr):
    real = abs(sympy.solve(f,x)[0].evalf())
    for i in range(itr):
        fx = f.subs(x,a).evalf()
        dfx = sympy.diff(f,x).subs(x,a).evalf()
        error = abs(real-a)
        print i,'\t',a,'\t',fx,'\t',dfx,'\t',error
        a -= fx/dfx

def bisectMethod(f, a, b, ep, itr):
    if a>=b or f.subs(x,a)*f.subs(x,b)>0:
        print 'INPUT ERROR'
        return
    c = (a+b)/2
    i = 0
    while b-c > ep and i<itr:
        i += 1
        fc = f.subs(x,c)
        error = abs(b-c)
        print i,'\t',a,'\t',b,'\t',c,'\t',fc,'\t',(b-c)
        if f.subs(x,a)*f.subs(x,b) <=0:
            a = c
        else:
            b = c
        c = (a+b)/2


# A = numpy.matrix([[0,1,2], [2,0,1], [1,4,1]], dtype=float)
# B = numpy.matrix([[1,1,-1], [1,2,-2], [-2,1,1]], dtype=float)
# C = numpy.matrix([[2,1,-1,-2], [4,4,1,3], [-6,-1,10,10], [-2,1,8,4]], dtype=float)

def kij(matrix):
    A = numpy.copy(matrix)
    (m,n) = A.shape
    for k in range(n):
        for i in range(k+1,n):
            A[i,k] = A[i,k]/A[k,k]
            for j in range(k+1,n):
                A[i,j] = A[i,j]-A[i,k]*A[k,j]
    U = A-numpy.tril(A,-1)
    L = numpy.tril(A,-1) + numpy.eye(n)
    return(L, U)

def kji(matrix):
    A = numpy.copy(matrix)
    (m,n) = A.shape
    P = numpy.eye(n)
    for k in range(n-1):
        m = k + numpy.argmax(abs(A[k:n,k]))     #GET MAX ROW
        P[:,[k,m]] = P[:,[m,k]]                 #SWAP COLS in P
        A[[k,m]] = A[[m,k]]                     #SWAP ROWS in A
        for i in range(k+1,n):
            A[i,k] = A[i,k]/A[k,k]
        for j in range(k+1,n):
            for i in range(k+1,n):
                A[i,j] = A[i,j]-A[i,k]*A[k,j]
    U = A-numpy.tril(A,-1)
    L = numpy.tril(A,-1) + numpy.eye(n)
    return(P,L,U)

# x = numpy.matrix([[1],[2],[-3]],dtype=float)
# b = numpy.matrix([[-4],[-1],[6]],dtype=float)

def solveLUxb(L,U,b):
    return numpy.linalg.inv(U).dot(numpy.linalg.inv(L).dot(b))
