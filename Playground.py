"""----------------------------------------------
-----------------Matt Bowyer---------------------
----------------------------------------------"""

import math
import sympy
import numpy




# x[n-1] = x[n] - f(x[n])/f'(x[n])



x = sympy.symbols('x', real=true)
func_1 = sympy.exp(x) * sympy.sin(x)
func_2 = (sympy.exp(x) - 1)/x
a = 1
n = 5
taylor_polynomial = sum([((x-a)**i) * sympy.diff(f,x,i).subs(x,a) /math.factorial(i) for i in range(n)] )

def taylor(f, a, n):
    derivatives = [sympy.diff(f,x,i).subs(x,a) for i in range(n)]
    factors = [ math.factorial(i) for i in range(n)]
    distances = [ (x-a)**i for i in range(n)
    expresions = [distances[i]*derivatives[i]/factors[i] for i in range(n)]
    taylor_poly = sum(expressions)





#A = numpy.matrix('1 2 3; 4 5 6; 7 8 9')
#A = numpy.matrix([[1,1,-1],[1,2,-2],[-2,1,1]])

a = [[1, 1, -1], [1, 2, -2], [−2, 1, 1]]

def kij(matrix):
    A = numpy.copy(matrix)
    (m,n) = A.shape
    count = 1
    print 'A = ',A.tolist()
    for k in range(n):
        for i in range(k+1,n):
            count += 1
            print 'A = ',A.tolist()
            A[i,k] = A[i,k]/A[k,k]
            for j in range(k+1,n):
                A[i,j] = A[i,j]-A[i,k]*A[k,j]
                count += 1
                print 'A = ',A.tolist()
    U = A-numpy.tril(A,-1)
    L = numpy.tril(A,-1) + numpy.eye(n)
    print 'L = ',A.tolist()
    print 'U = ',A.tolist()
    return(A, L, U)

'''
n=input('n=')
a=ones(n,n)+n*eye(n,n);
A=a; tstart=tic
[L U ] = kij( a )
telapsed=toc(tstart)
where the function is defined by:
function [L U ] = kij( a )
m=size(a);
n=m(1);
for k=1:n
for i=k+1:n
a(i,k)=a(i,k)/a(k,k);
for j=k+1:n
a(i,j)=a(i,j)-a(i,k)*a(k,j);
end
end
end
L = tril(a,-1);
U=a-L;
L=L+eye(n);
End
'''



'''  Approximations and Error


#1
e = 1 - 3*(4/3 - 1),
e =2.2204e-16
=> 1/3 is rounded to 0.33333333326

#2
a = 0.0;
for i = 1:10
 a = a + 0.1;
end
a == 1
ans = 0

#
b = 1e-16 + 1 - 1e-16;
c = 1e-16 - 1e-16 + 1;
b == c
ans = 0

Single: SEEEEEEE EMMMMMMM MMMMMMMM MMMMMMMM
Double: SEEEEEEE EEEEMMMM MMMMMMMM MMMMMMMM MMMMMMMM MMMMMMMM MMMMMMMM MMMMMMMM

S = 1s 8e 21m
D = 1s 11e 51m

0...pi/2 in 10 steps = 0...10*math.pi/20

r = [math.pi**(-i) for i in range(11)]
sin = [math.sin(i) for i in r]
sin_id = [math.sqrt(1- math.cos(i)**2) for i in r]
for i in range(11):
    print i,'\t',sin[i],'\t',sin_id[i],'\t', abs(sin[i]-sin_id[i])
