"""----------------------------------------------
-----------------Matt Bowyer---------------------
----------------------------------------------"""

import math
import sympy
import numpy
import scipy
import matplotlib.pyplot as plt

n = 2
sub_x = sympy.symbols('x')
a = -4
b = 4
steps = range(n)
func = sympy.cos(sub_x)
steps = range(n)

x_norm = [a+(i+1.0)*(b-a)/n for i in steps]
y_norm = [func.subs(sub_x,x) for x in x_norm]
p_norm = numpy.polyval(numpy.polyfit(x_norm, y_norm, n-1), x_norm)
e_norm = [y_norm[i]-p_norm[i] for i in steps]

x_cheb = [((b-a)/2)*math.cos(math.pi*(2*i+1)/(2*n+2))+((a+b)/2) for i in steps][::-1]
y_cheb = [func.subs(sub_x,x) for x in x_cheb]
p_cheb = numpy.polyval(numpy.polyfit(x_cheb, y_cheb, n-1), x_cheb)
e_cheb = [y_cheb[i]-p_cheb[i] for i in steps]

sub_x = sympy.symbols('x')
n = 15
a = -4
b = 4
steps = range(n)
func = 1.0/(1+10*sub_x**2)
steps = range(n)

x_norm = [a+(i+1.0)*(b-a)/n for i in steps]
y_norm = [func.subs(sub_x,x) for x in x_norm]
p_norm = numpy.polyval(numpy.polyfit(x_norm, y_norm, n-1), x_norm)
e_norm = [y_norm[i]-p_norm[i] for i in steps]

x_cheb = [((b-a)/2)*math.cos(math.pi*(2*i+1)/(2*n+2))+((a+b)/2) for i in steps][::-1]
y_cheb = [func.subs(sub_x,x) for x in x_cheb]
p_cheb = numpy.polyval(numpy.polyfit(x_cheb, y_cheb, n-1), x_cheb)
e_cheb = [y_cheb[i]-p_cheb[i] for i in steps]

plt.plot(x_norm, y_norm, 'g-', label = 'Function 1')
plt.plot(x_norm, p_norm, 'bo', label = 'Interpolation 2')
plt.plot(x_norm, e_norm, 'r^', label = 'Error 3' )

plt.plot(x_cheb, y_cheb, 'g-', label = 'Function 1')
plt.plot(x_cheb, p_cheb, 'bo', label = 'Interpolation 2')
plt.plot(x_cheb, e_cheb, 'b^', label = 'Error 3' )
plt.show()




sub_x = sympy.symbols('x')
n = 2
a = -4
b = 4
steps = range(n)
func = sympy.cos(sub_x)
steps = range(n)
x_cheb = [((b-a)/2)*math.cos(math.pi*(2*i+1)/(2*n+2))+((a+b)/2) for i in steps][::-1]
y_cheb = [func.subs(sub_x,x) for x in x_cheb]
p_cheb = numpy.polyval(numpy.polyfit(x_cheb, y_cheb, n-1), x_cheb)
e_cheb = [y_cheb[i]-p_cheb[i] for i in steps]
plt.plot(x_cheb, y_cheb, 'g-', label = 'Function 1')
plt.plot(x_cheb, p_cheb, 'bo', label = 'Interpolation 2')
plt.plot(x_cheb, e_cheb, 'r^', label = 'Error 3' )
plt.show()



n = 10
sub_x = sympy.symbols('x')
a = -4
b = 4
steps = range(n)
func = 1.0/(1+10*sub_x**2)
steps = range(n)
x_norm = [a+(i+1.0)*(b-a)/n for i in steps]
y_norm = [func.subs(sub_x,x) for x in x_norm]
p_norm = numpy.polyval(numpy.polyfit(x_norm, y_norm, n-1), x_norm)
e_norm = [y_norm[i]-p_norm[i] for i in steps]
plt.plot(x_norm, y_norm, 'g-', label = 'Function 1')
plt.plot(x_norm, p_norm, 'bo', label = 'Interpolation 2')
plt.plot(x_norm, e_norm, 'r^', label = 'Error 3' )
plt.show()

# Make matrices print fractions
# import fractions
# import numpy
# numpy.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})
# numpy.set_printoptions(formatter={'all':lambda x: str(("%.4f" % round(x,4)))})
# numpy.set_printoptions(formatter={'all':lambda x: str(x)})

'''
for t in range(-1,len(T)):
    if(t==-1):
        print 'i= \tx[0]= \tx[1]= \tx[2]= \te= \tr='
    else:
        print str(t),'\t',T[t]

'''

out = [[str(("%.4f" % round(T[m,n],4))) for n in range(T.shape[1])] for m in range(T.shape[0])]

def seidel(itr, A, b, x):
    X = [numpy.matrix([[0.0] for i in range(len(b))]) for k in range(itr+1)]
    E = [numpy.amax(abs(x)) for i in range(itr+1)]
    R = [0.0 for i in range(itr+1)]
    for k in range(itr):
        for i in range(len(b)):
            left = sum([ (A[i,n] * X[k+1][n]) for n in range(0,i)])
            right = sum([ (A[i,n] * X[k][n]) for n in range(i+1,len(b))])
            X[k+1][i,0] = 1.0 * (b[i,0] - left - right) / A[i,i]
            E[k+1] = numpy.amax(abs(X[k+1] - x))
            R[k+1] = E[k+1]/ E[k]
    T = numpy.matrix([X[i].ravel().tolist()[0] + [E[i]] + [R[i]] for i in range(len(X))])
    trials = [[str(("%.4f" % round(T[m,n],4))) for n in range(T.shape[1])] for m in range(T.shape[0])]
    return (X,E,R,T)

def jacobi(itr, A, b, x):
    X = [numpy.matrix([[0.0] for i in range(len(b))]) for k in range(itr+1)]
    E = [numpy.amax(abs(x)) for i in range(itr+1)]
    R = [0.0 for i in range(itr+1)]
    D = numpy.tril(A) - numpy.tril(A,-1)
    I = numpy.linalg.inv(D)
    C = I.dot(b)
    F = I.dot(A - D)
    for k in range(itr):
        X[k+1] = C - F.dot(X[k])
        E[k+1] = numpy.amax(abs(X[k+1] - x))
        R[k+1] = E[k+1]/ E[k]
    T = numpy.matrix([X[i].ravel().tolist()[0] + [E[i]] + [R[i]] for i in range(len(X))])
    trials = [[str(("%.4f" % round(T[m,n],4))) for n in range(T.shape[1])] for m in range(T.shape[0])]
    return (X,E,R,T)


# A = numpy.matrix('9 1 1 1; 1 8 1 1; 1 1 7 1; 1 1 1 6')
# b = numpy.matrix('75; 54; 43; 34')


# A = numpy.matrix([[1.0/2, -1.0/4], [-1.0/4, 1.0/2]])
# b = numpy.matrix([[3], [-3]])
# x = numpy.matrix([[4], [-4]])


# A = numpy.matrix([[9,1,1], [2,10,3], [3,4,11]])
# b = numpy.matrix([[10], [19], [0]])
# x = numpy.matrix([[1], [2], [-1]])

# A = numpy.matrix('4 -1 0 -1 0 0 0 0 0; -1 4 -1 0 -1 0 0 0 0; 0 -1 4 0 0 -1 0 0 0;  -1 0 0 4 -1 0 -1 0 0;  0 -1 0 -1 4 -1 0 -1 0;  0 0 -1 0 -1 4 0 0 -1;  0 0 0 -1 0 0 4 -1 0;  0 0 0 0 -1 0 -1 4 -1;  0 0 0 0 0 -1 0 -1 4')
# b = numpy.matrix('4; -1; -5; -2; 2; 2; -1; 1; 6')
# x = numpy.matrix('1; 0; -1; 0; 1; 1; 0; 1; 2')

𝐴𝑥 = 𝑏
==> [𝐷 + (𝐴 − 𝐷)]𝑥 = 𝑏
==> 𝐷𝑥 = 𝑏 − (𝐴 − 𝐷)𝑥
==> 𝑥 = 𝐷^(−1)𝑏 − 𝐷^(−1)(𝐴 − 𝐷)𝑥
so we can derive an iterative method:
𝑥^(𝑘+1) = 𝐷^(−1)𝑏 − 𝐷^(−1)(𝐴 − 𝐷)𝑥[𝑘]

Write a matlab program that implements this method using
Matlab matrix definitions e.g.
𝐷 = 𝑡𝑟𝑖𝑙(𝐴) − 𝑡𝑟𝑖𝑙(𝐴, −1);
𝑥(𝑘 + 1) = 𝑖𝑛𝑣(𝐷) ∗ (𝑏 − (𝐴 − 𝐷) ∗ 𝑥(𝑘)),
you can use as stopping criterion
𝑛𝑜𝑟𝑚(𝑥(𝑘 + 1) − 𝑥(𝑘), 𝛾),
𝛾 = 2, 𝑜𝑟
𝛾 = 1, 𝑜𝑟
𝛾 = 𝑖𝑛𝑓
for the 3 norms that we learned in class,
the 2 norm or
the 1 norm or
the infinity or
maximum norm


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
