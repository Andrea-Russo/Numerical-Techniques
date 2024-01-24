# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 19:01:08 2023

@author: andre

The Metropolis–Hastings algorithm can draw samples from any probability distribution
with probability density P(x), provided that we know a function f(x) proportional
to the density P and the values of f(x) can be calculated.
The requirement that f(x) must only be proportional to the density,
rather than exactly equal to it, makes the Metropolis–Hastings algorithm particularly
useful, because calculating the necessary normalization factor is often extremely
difficult in practice.

The Metropolis–Hastings algorithm works by generating a sequence of sample values
in such a way that, as more and more sample values are produced, the distribution
of values more closely approximates the desired distribution.
These sample values are produced iteratively, with the distribution of the next sample
being dependent only on the current sample value (thus making the sequence of samples into a Markov chain).
Specifically, at each iteration, the algorithm picks a candidate for the next sample
value based on the current sample value. Then, with some probability,
the candidate is either accepted (in which case the candidate value is used in the next iteration)
or rejected (in which case the candidate value is discarded, and current value is reused
in the next iteration)—the probability of acceptance is determined by comparing the values
of the function f(x) of the current and candidate sample values with respect to the desired distribution.
"""

"""Suppose we want to end up sampling from the nomal distribution p(x)=1/sqrt{\pi}exp(-(x-1)^2), 
a Gaussian with variance=0.5 and mean=1. We don't know the normalisation but we know it to be proportional to
g(x)=exp(-(x-1)^2)"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd

def f(x):  # f can be any probability distribution proportional to the probability distribution of interest
    return np.exp(-(x-3)**2)

#First we draw a sample , 
#  and initialise
x0=rnd.uniform(-10,10)
N=100000
samples=np.arange(N,dtype=np.float)
samples[0]=x0
#Then we generate a candidate from a distribution q(x_t|x_{t-1}) and test to see if we accept or reject the new value
# accoridng to the Metropolis-Hastings algorithm. 
# In this case we choose a normal distribution centered at x_{t-1}.
for t in range(0,N-1):
    xt=rnd.normal(loc=samples[t],size=1) #generate candidate
    alpha=f(xt)/f(samples[t])   #compute acceptance ratio as quotient of distribution
    u=rnd.uniform()
    if u<=alpha:    # accept new point with probability alpha
        samples[t+1]=xt
    else:
        samples[t+1]=samples[t]

print(samples)

plt.hist(samples[1000:],bins=100) #plot samples eliminagting the first 1000 to only get convergent data




