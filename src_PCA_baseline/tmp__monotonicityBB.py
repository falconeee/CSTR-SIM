from sympy import *
var('a c d A B')

A = Matrix([[1, 0], [a, c]])
A

B = Matrix([[1, d], [0, 1]])

M = A.multiply(B)
M


var('x1, x2, x3, x4, x5, x6' )

var('x01, x02, x03, x04, x05, x06' )
var('xm1, xm2, xm3, xm4, xm5, xm6' )

var('s1 s2 s3 s4 s5 s6')
C = diag(s1**2, s2**2, s3**2, s4**2, s5**2, s6**2)
print('type=', type(C), 'shape=', C.shape, 'C='); pprint(C)
Cinv = C**(-1)
print('Cinv='); pprint(Cinv)

x = Matrix([x1, x2, x3, x4, x5, x6])
s = Matrix([s1, s2, s3, s4, s5, s6])
idxo = (0, 4, 3, 1)
idxm = (5, 2)
idx = idxo + idxm
#xo = Matrix([x1, x4, x3, x2])
xo = x.row(idxo)
#xm = Matrix([x5, x6])
xm = x.row(idxm)

xtil = x.row(idx)
print('xtil='); pprint(xtil.T)


def EM2(x, s, idxo, idxm):
    xo = x.row(idxo)
    xm = x.row(idxm)
    no = len(xo)
    nm = len(xm)
    n = no + nm
    print('no=', no, 'nm=', nm, 'n=', n)
    
    print('x='); pprint(x.T)

    xtil = xm.row_insert(0, xo)
    print('xo='); pprint(xo.T)
    print('xm='); pprint(xm.T)
    print('xtil='); pprint(xtil.T)
    
    so = s.row(idxo)
    sm = s.row(idxm)
    print('so='); pprint(so.T)
    print('sm='); pprint(sm.T)

    C = zeros(n,n)

    
    so2 = matrix_multiply_elementwise(so, so)
    Coo = zeros(no, no)
    for i in range(no):
        Coo[i,i] = so2[i]
        C[i,i] = so2[i]
        print('so2='); pprint(so2.T)
    print(type(Coo))
    print('Coo=', 'shape=', Coo.shape, ); pprint(Coo)

    sm2 = matrix_multiply_elementwise(sm, sm)
    Cmm = zeros(nm, nm)
    for i in range(nm):
        Cmm[i,i] = sm2[i]
        C[i+no,i+no] = sm2[i]
        print('sm2='); pprint(sm2.T)
    print(type(Cmm))
    print('Cmm=', 'shape=', Cmm.shape); pprint(Cmm)
    
    print('C=', 'shape=', C.shape); pprint(C)
    for i in range(no):
        C[i,i] = Coo[i]


EM2(x, s, idxo, idxm)