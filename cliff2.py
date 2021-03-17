import numpy as np;
import math, cmath;
from itertools import product;
from numpy.linalg import matrix_power;
from functools import reduce;
import numpy.linalg as la;
import multiprocessing as mp;
import more_itertools;
import time;



### Code specific to n = 2 ######################################################
# Replace in cliff.py everything below the part specific to n = 1

# Given matrices U1,U2 (of a conjugate tuple), returns an eigenvector of eigenvalue 1
def EvecOne2(Us):
  U1,U2 = Us
  proj1 = (1/d) * sum([matrix_power(U1,k) for k in Zd])
  proj2 = (1/d) * sum([matrix_power(U2,k) for k in Zd])
  eigs = la.eig(proj1 @ proj2)
  return StandV(np.reshape(eigs[1][:,np.where(eigs[0].round(10) == 1)], (d**2,1)))

# From a conjugate tuple, returns the unitary conjugating basic Paulis to it
def UfromCzCx2(CzCx):
  #assert mequal(CzCx[0][0] @ CzCx[0][1], w*CzCx[0][1] @ CzCx[0][0])
  #assert mequal(CzCx[1][0] @ CzCx[1][1], w*CzCx[1][1] @ CzCx[1][0])
  #assert commPair(CzCx)
  Ket0 = EvecOne2((CzCx[0][0],CzCx[1][0]))
  V1 = CzCx[0][1]
  V2 = CzCx[1][1]
  x = np.array([(matrix_power(V1,a) @ matrix_power(V2,b) @ Ket0).reshape(9)  for a in Zd for b in Zd])
  return np.column_stack(x)

# Given a list of kth-level gates (up to phase), returns all conjugate pairs (up to phase) 
def nextLevelAsPairs2(Ck):
  fxOrdDs = pool_filter(orderD, [fixPhase(U) for U in Ck])
  return  pool_filter_lazy(hasRelPhase,product(fxOrdDs,fxOrdDs))

# Check if two matrices commute
def commute(m1,m2):
  return mequal(m1@m2,m2@m1)

# Given two conjugate pairs, check if they form a conjugate tuple (for n = 2)
def commPair(tpl):
  p1,p2 = tpl
  return commute(p1[0],p2[0]) and commute(p1[0],p2[1]) and commute(p1[1],p2[0]) and commute(p1[1],p2[1])

# Given a list of kth-level gates (up to phase), returns all conjugate tuples (for n = 2, up to phase)
def nextLevelAsTuples2(Ck):
  conjPairs = nextLevelAsPairs2(Ck)
  print(len(conjPairs), "conjugate pairs")
  return  pool_filter_lazy(commPair,product(conjPairs,conjPairs))

# Given a list of all conjugate tuples (up to phase) of kth-level gates, returns all (k+1)th-level gates (up to phase)
def tuplesToNextLevel2(cPairs):
    return pool_map(UfromCzCx2,[((w**k1 * cPair[0][0], w**k2 * cPair[0][1]),(w**k3 * cPair[1][0], w**k4 * cPair[1][1])) for cPair in cPairs for k1 in Zd for k2 in Zd for k3 in Zd for k4 in Zd])
    
# Symplectic inner product for n = 2
def SP2(pp):
  return (pp[0][0]*pp[1][1] - pp[0][1]*pp[1][0] + pp[0][2]*pp[1][3] - pp[0][3]*pp[1][2]) % d == 0
 
# Non-zero phase-points up to scalar multiples
SCtuples2 = [(1,a,b,c) for a in Zd for b in Zd for c in Zd] + [(0,1,b,c) for b in Zd for c in Zd] + [(0,0,1,c) for c in Zd] + [(0,0,0,1)]

# All Lagrangian semi-bases
LagBases = list(filter(SP2, product(SCtuples2,SCtuples2)))

# Check if a conjugate pair defines a semi-Clifford gate
def isSC2(cPairs):
  return any([isPauli(matrix_power(cPairs[0][0],k[0]) @ matrix_power(cPairs[0][1],k[1]) @ matrix_power(cPairs[1][0],k[2]) @ matrix_power(cPairs[1][1],k[3])) for k in LagBases])
    
#################################################################################
  
  
  
# Main program
if __name__ == "__main__":
    numCPUs = mp.cpu_count();
    print("Number of CPUs:", numCPUs)
    
    C2ap = nextLevelAsTuples2(nPaulis)                                          # Generate Cliffords (up to phase) as conjugate tuples of Paulis
    print(".")
    print("C2 has ", len(C2ap), "* d^(2n) elements (up to phase)")
    
    with open('n2-c2-d' + str(d) + '-asAP.npy', 'wb') as f:                     # Save data file of C3 gates
    np.save(f, C2ap)
    
    C2 = [UfromCzCx(cPair) for cPair in conjPairs]                              # Generate Cliffords (up to phase)    

    tic = time.perf_counter()
    C3ap = nextLevelAsTuples2(C2)                                               # Generate C3 (up to phase) as conjugate tuples of Cliffords
    print("C3 has ", len(C3ap), " elements (up to phase)")
    toc = time.perf_counter()
    print(f"{toc - tic:0.4f} seconds")
    
    tic = time.perf_counter()
    if all(pool_map(isSC2, C3ap)):                                              # Check if all C3 gates are semi-Clifford
        print("All C3 are semi-Clifford")
    else:
        print("NOT ALL are semi-Clifford")
    toc = time.perf_counter()
    print(f"{toc - tic:0.4f} seconds")
