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


#a = numpy.matrix([[0,1,2], [2,0,1], [1,4,1]], dtype=float)
#b = numpy.matrix([[1,1,-1], [1,2,-2], [-2,1,1]], dtype=float)
#c = numpy.matrix([[2,1,-1,-2], [4,4,1,3], [-6,-1,10,10], [-2,1,8,4]], dtype=float)

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
    piv = range(n)
    permutations = 0
    for k in range(n-1):
        m = k + numpy.argmax(abs(A[k:n,k]))     #GET MAX ROW
        piv[k], piv[m] = piv[m], piv[k]         #SAVE SWAP in PIV
        A[[k,m]] = A[[m,k]]                     #SWAP ROWS
        print k,'\t',m,'\n',A
        for i in range(k+1,n):
            A[i,k] = A[i,k]/A[k,k]
        for j in range(k+1,n):
            for i in range(k+1,n):
                A[i,j] = A[i,j]-A[i,k]*A[k,j]
    U = A-numpy.tril(A,-1)
    L = numpy.tril(A,-1) + numpy.eye(n)
    P = numpy.eye(n)[:,piv]
    return(P,L,U)
