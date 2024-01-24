"""
Metropolis-Hasting Algorithm

Samnpling a lognormal distribution with Metropolis-Hasting
f(x) = (1/x sqrt(2 pi sigma^2))exp(- (ln(x)-mu)^2 /2*sigma^2)

"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd

# Set parameters of desired distribution
mu = 0
sigma = 0.5  # hence variance of 0.5

size = 100000
x = np.zeros(size)

x_t = abs(rnd.uniform(-10, 10)) # As lognormal only handles positive values
x[0] = x_t

for i in range(1,size):
    # Sample new point (choose symmetrical distribution for simplicity)
    x_t1 = abs(rnd.normal(x_t ,1))
    
    # Compute acceptance probability
    rf = x_t/x_t1 * np.exp((- (np.log(x_t1) - mu)**2 + (np.log(x_t) - mu)**2) / (2 * sigma**2) )
    A = min(1, rf)
    
    # Accept or reject point
    if rnd.uniform(0, 1) <= A:
        x_t = x_t1
        x[i] = x_t1
    else:
        x[i] = x_t  # Store current state if rejected
    
plt.hist(x[int(size*0.1):], bins=100, density=True)
plt.xlabel('x')
plt.ylabel('Frequency')
plt.title('Metropolis-Hastings Sampling')
plt.show() 