from itertools import product;
import more_itertools;
import multiprocessing as mp;
import time;
import numpy as np;



d = 3;                                                                          # Qudit dimension (prime, e.g. 2, 3, 5...)
Zd = list(range(d));                                                            # {0,...,d}


# Given a pair of septuples of integers, whether they define a conjugate pair
def isConj(pair):
  d1, d2, d3, k1, j1, k2, j2 = pair[0]; 
  e1, e2, e3, a1, b1, a2, b2 = pair[1];
  return (-2*e1*j1 - e3*j2 + 2*d1*b1 + d3*b2) % 3 == 0 and (-e3*j1 - 2*e2*j2 + d3*b1 + 2*d2*b2) % 3 == 0 and (e1*j1**2  - a1*j1 + e2*j2**2 - a2*j2 + e3*j1*j2 - d1*b1**2 + k1*b1 - d2*b2**2 + k2*b2 - d3*b1*b2) % 3 == 1

# Given two two-qutrit Clifford gates as integers, whether they commute
def isComm(pair0,pair1):
  d1, d2, d3, k1, j1, k2, j2 = pair0; 
  e1, e2, e3, a1, b1, a2, b2 = pair1;
  return (-2*e1*j1 - e3*j2 + 2*d1*b1 + d3*b2) % 3 == 0 and (-e3*j1 - 2*e2*j2 + d3*b1 + 2*d2*b2) % 3 == 0 and (e1*j1**2  - a1*j1 + e2*j2**2 - a2*j2 + e3*j1*j2 - d1*b1**2 + k1*b1 - d2*b2**2 + k2*b2 - d3*b1*b2) % 3 == 0

# Checks if a pair of conjugate pairs is a conjugate tuple
def isConjTple(tple):
  return isComm(tple[0][0],tple[1][0]) and isComm(tple[0][0],tple[1][1]) and isComm(tple[0][1],tple[1][0]) and isComm(tple[0][1],tple[1][1]) 
  
# Symplectic inner product for n = 2
def SP2(pp):
  return (pp[0][0]*pp[1][1] - pp[0][1]*pp[1][0] + pp[0][2]*pp[1][3] - pp[0][3]*pp[1][2]) % d == 0
 
# Non-zero phase-points up to scalar multiples
SCtuples2 = [(1,a,b,c) for a in Zd for b in Zd for c in Zd] + [(0,1,b,c) for b in Zd for c in Zd] + [(0,0,1,c) for c in Zd] + [(0,0,0,1)]

# All Lagrangian semi-bases
LagBases = list(filter(SP2, product(SCtuples2,SCtuples2)))

# Check if a conjugate tuple and a Lagrangian semi-basis generates a Pauli gate
def makesPauli(ct, lb):
  return all((np.array(ct[0][0][0:3])*lb[0][0] + np.array(ct[0][1][0:3])*lb[0][1] + np.array(ct[1][0][0:3])*lb[0][2] + np.array(ct[1][1][0:3])*lb[0][3]) % 3 == [0,0,0]) and all((np.array(ct[0][0][0:3])*lb[1][0] + np.array(ct[0][1][0:3])*lb[1][1] + np.array(ct[1][0][0:3])*lb[1][2] + np.array(ct[1][1][0:3])*lb[1][3]) % 3 == [0,0,0])
  
# Check if a conjugate tuple defines a semi-Clifford gate
def isSC2(ct):
  return any([makesPauli(ct,k) for k in LagBases])



# Main program
if __name__ == "__main__":
    # Generate all conjugate pairs as pairs of septuples of integers
    cp = list(filter(isConj, list(product(product(range(3), repeat=7), repeat=2))));

    print(len(cp));
    
    tic = time.perf_counter()

    # Generate all conjugate tuples from conjugate pairs
    ct = list(filter(isConjTple, product(cp, repeat=2)));
    print(len(ct));

    toc = time.perf_counter()
    print(f"{toc - tic:0.4f} seconds")
    
    with open('ct.npy', 'wb') as f:                                             # Save data file of C3 gates
        np.save(f, ct)

    # Check if all conjugate tuples are semi-Clifford
    print(all([isSC2(x) for x in ct]))