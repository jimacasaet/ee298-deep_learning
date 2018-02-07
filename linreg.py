'''
NABUS, Martin Roy E.
ASSIGNMENT 1: Gradient Descent for Linear Regression of Polynomial Function
Answer to learning rate question is at the bottom of the code
'''

import sys
import time
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

time_start = time.time()

args = len(sys.argv)
if(args < 2 or args > 5) : print('Invalid number of arguments'); sys.exit()

# Obtain coefficients from command line/terminal input
C0 = 0; C1 = 0; C2 = 0; C3 = 0
if(args == 2) : C0 = float(sys.argv[1])
elif(args == 3) : C1 = float(sys.argv[1]); C0 = float(sys.argv[2])
elif(args == 4) : C2 = float(sys.argv[1]); C1 = float(sys.argv[2]); C0 = float(sys.argv[3])
else : C3 = float(sys.argv[1]);  C2 = float(sys.argv[2]); C1 = float(sys.argv[3]); C0 = float(sys.argv[4]) # args == 5

# Set A & B arrays (Ax = B)
# For quadratic & cubic fxns, the x-coordinates of the critical points are derived to obtain an interval for the input values that allows the resulting graph to exhibit the changes in rate and/or concavity of the user-defined function
points = 50 # Number of points used; also the number of rows in A
A0 = np.ones(points) # Initialize rightmost column of A for CONSTANT, LINEAR, QUADRATIC, and CUBIC functions
if(args == 3) : A1 = np.linspace(-10,10,num=points) # Initialize left column of A for LINEAR fxns
elif(args == 4) : # Initialize columns of A (except rightmost) for QUADRATIC fxns
	h = C1/(-C2) # x-coordinate of extremum
	p = 1/(4*C2) # distance between focus and extremum
	A1 = np.linspace(h-20*p,h+20*p,num=points) 
	A2 = np.power(A1,2)
elif(args == 5) : # Initialize columns of A (except rightmost) for CUBIC fxns
	p = np.poly1d([C3,C2,C1,C0])
	p = p.deriv()
	rs = np.unique(p.roots[np.isreal(p.roots)]) # x-coordinates of critical point/s
	if(len(rs) == 2) : # two critical points (extrema)
		d = (rs[1] - rs[0])/2
		A1 = np.linspace(rs[0]-2*d,rs[1]+2*d,num=points)
	elif(len(rs) == 1) : A1 = np.linspace(rs[0]-5,rs[0]+5,num=points) # one critical point (saddle)
	else:	# no critical points; derive inflection point instead
		p = p.deriv()
		rs = p.roots
		A1 = np.linspace(rs[0]-5,rs[0]+5,num=points)
	A2 = np.power(A1,2)
	A3 = np.power(A1,3)
if(args == 2) : B = C0*A0
elif(args == 3) : B = C1*A1 + C0*A0
elif(args == 4) : B = C2*A2 + C1*A1 + C0*A0
else : B = C3*A3 + C2*A2 + C1*A1 + C0*A0 # args == 5

# B0 = B # preserve original B if you want to plot graph
B = B + np.random.uniform(-1,1,size=points) # Apply uniform noise

# Change row vectors to column vectors
A0 = A0.reshape(-1,1)
if(args > 2) : A1 = A1.reshape(-1,1)
if(args > 3) : A2 = A2.reshape(-1,1)
if(args > 4) : A3 = A3.reshape(-1,1)
B = B.reshape(-1,1)

# plt.plot(A1, B0, 'b-', A1, B, 'ro') # Replace A1 w/ A0 for constant graph

# Concatenate A column vectors to form the A-matrix
if(args > 4) : A2 = np.concatenate((A3,A2),1)
if(args > 3) : A1 = np.concatenate((A2,A1),1)
if(args > 2) : A0 = np.concatenate((A1,A0),1)

# Gradient Descent
AT = A0.transpose()
x = np.zeros(args-1) # initial prediction is all zeros
x = x.reshape(-1,1)
eps = 0.0000075	# epsilon
delta = 0.00001	# delta is fixed to 1e-5
M = np.matmul(np.matmul(AT,A0),x) - np.matmul(AT,B) # (A^T)Ax - (A^T)B
err = la.norm(M)
while(err > delta) : # Recursive part of gradient descent
	x = x - eps*(M)
	M = np.matmul(np.matmul(AT,A0),x) - np.matmul(AT,B)
	err = la.norm(M)
print(x)

# Get RMSE between results and actual coefficients
if(args == 2) : orig = np.array([C0])
elif(args == 3) : orig = np.array([C1,C0])
elif(args == 4) : orig = np.array([C2,C1,C0])
else : orig = np.array([C3,C2,C1,C0])
orig = orig.reshape(-1,1)
print('RMSE =',la.norm(x - orig))

time_run = time.time() - time_start
print('Time = ',time_run,'sec')

# plt.show()

'''
Q: What is the recommended learning rate?
A: From my runs using different polynomial equations and different values of epsilon (1e-1 or 1x10^-1, 7.5e-2, 5e-2, 2.5e-2, 1e-2, 7.5e-3, ..., 1e-7) for each equation, I observed the following:
	1. Using a higher-valued epsilon may cause the result to diverge towards infinity, but is generally faster if it manages to converge to the correct result. Conversely, a lower-valued epsilon will most likely converge to the correct result, but is generally slower than a higher-valued epsilon
	2. The following were the median best values (i.e. managed to get a result with delta < 1e-5 in the least amount of time) of epsilon for the following nth degree polynomials, regardless of the nature of the graph (e.g. number of critical points)
		a. Constant: 1e-2 
		b. Linear: 1e-3
		c. Quadratic: 1e-4
		d. Cubic: 1e-5
		* RMSE between results and actual coefficients ranged from 0.03 to 0.2
From these results, it would seem that the best epsilon used in the basic gradient descent algorithm for linear regression of an nth degree polynomial function is 1e-(n+2) or 1 x 10^-(n+2) assuming delta = 0.00001; however, tests for higher-degree polynomials may be required to verify this observation
	* Hence, for this assignment, the best epsilon to use is 0.00001 since the code accepts up to 3rd degree polynomials. Safer alternatives are 0.0000075 and 0.000005.
'''


