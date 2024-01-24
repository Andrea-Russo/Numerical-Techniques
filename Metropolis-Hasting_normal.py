"""
Metropolis-Hasting Algorithm

As a first exercise, let us try to sample a gaussian distribution N(5,4)
The only thig we know is f(x)=exp(-(x-5)^2/8), we are missing the normalisaiton
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd

"""
# 1) sample a starting point x_t
x_t = rnd.uniform(-10,10)

# 2) sample x_t from N(x,1)
x_t1 = rnd.normal(x_t,1)

# 3) compute r_r, r_g and A(x_t -> x_t1)
rf = np.exp(-(x_t1 - 5)**2 / 8) / np.exp(-(x_t - 5)**2 / 8)

rg = np.exp((x_t1)**2 / 2) / np.exp((x_t)**2 / 2)

A = min(1, rf * rg)
"""

size = 10000
x = np.zeros(size)

# Initial state chosen uniformly at random between -10 and 10
x_t = rnd.uniform(-10,10)
x[0] = x_t

for i in range(1,size):
    # Propose a new state x_t1 based on the current state x_t
    x_t1 = rnd.normal(x_t,1)
    
    # Compute the acceptance ratio
    rf = np.exp(-(x_t1 - 5)**2 / 8) / np.exp(-(x_t - 5)**2 / 8)
    rg = np.exp(-(x_t1 - x_t)**2 / 2) / np.exp(-(x_t - x_t1)**2 / 2)
    A = min(1, rf)
    
    # Accept or reject point
    if rnd.uniform(0, 1) < A:
        x_t = x_t1
        x[i] = x_t1
    else:
        x[i] = x_t  # Store current state if rejected
        
# Plotting the histogram of the samples, excluding the initial burn-in period
plt.hist(x[int(size*0.1):], bins=40, density=True)
plt.xlabel('x')
plt.ylabel('Frequency')
plt.title('Metropolis-Hastings Sampling')
plt.show()