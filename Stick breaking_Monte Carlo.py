"""
Here, we prove the famous Stick Breaking problem witha Monte Carlo simulation.
If a stick of unit lenght is broken in three parts uniform at random,
what is the probability that you could form a triangle with the three pieces?

"""
# Import libraries
import numpy.random as rnd

# Set number of Monte Carlo simulations
samples = 10000

# Generate breaking points of the stick
X = rnd.uniform(0, 1, samples)
Y = rnd.uniform(0, 1, samples)

# This variable counts the numbe rof triangles formed
triangle = 0

# Monte carlo simulation
for i in range(samples):
    
    # It is important to set the correct min/max variable as L1 and L3
    L1 = min(X[i], Y[i])
    L2 = abs(Y[i] - X[i])
    L3 = 1 - max(X[i], Y[i])
    
    # If it is a triangle, count it
    if (L1 + L2 > L3) and  (L1 + L3 > L2) and (L2 + L3 > L1):
        triangle += 1

# Compute probability of triangle forming
prob = round(triangle/samples,3)
    
print(f"The probability of getting a triangle is: {prob*100}%")