import numpy

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
